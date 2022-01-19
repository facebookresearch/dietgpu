/**
 * Copyright 2004-present Facebook. All Rights Reserved.
 */

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/PtxUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

#include <glog/logging.h>
#include <cmath>
#include <cub/block/block_scan.cuh>
#include <memory>
#include <vector>

namespace dietgpu {

using TableT = uint32_t;

// We are limited to 11 bits of probability resolution
// (worst case, prec = 12, pdf == 2^12, single symbol. 2^12 cannot be
// represented in 12 bits)
inline __device__ TableT
packDecodeLookup(uint32_t sym, uint32_t pdf, uint32_t cdf) {
  static_assert(sizeof(ANSDecodedT) == 1, "");
  // [31:20] cdf
  // [19:8] pdf
  // [7:0] symbol
  return (cdf << 20) | (pdf << 8) | sym;
}

inline __device__ void
unpackDecodeLookup(TableT v, uint32_t& sym, uint32_t& pdf, uint32_t& cdf) {
  // [31:20] cdf
  // [19:8] pdf
  // [7:0] symbol
  sym = v & 0xffU;
  v >>= 8;
  pdf = v & 0xfffU;
  v >>= 12;
  cdf = v;
}

template <int ProbBits>
__device__ void decodeOneWarp(
    ANSStateT& state,

    // Start offset where this warp is reading from the
    // compressed input. As a variable number of lanes
    // wish to read from the compressed offset each
    // iteration, this offset upon calling is one after
    // the last offset, if any, this warp will be reading rom.
    uint32_t compressedOffset,

    const ANSEncodedT* __restrict__ in,

    // Shared memory LUTs
    const TableT* lookup,

    // Output: number of words read from compressed input
    uint32_t& outNumRead,

    // Output: decoded symbol for this iteration
    ANSDecodedT& outSym) {
  constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);

  auto s_bar = state & StateMask;

  uint32_t sym;
  uint32_t pdf;
  uint32_t sMinusCdf;
  unpackDecodeLookup(lookup[s_bar], sym, pdf, sMinusCdf);

  // We always write a decoded value
  outSym = sym;
  state = pdf * (state >> ProbBits) + ANSStateT(sMinusCdf);

  // We only sometimes read a new encoded value
  bool read = state < kANSMinState;
  auto vote = __ballot_sync(0xffffffff, read);
  // We are reading in the same order as we wrote, except by decrementing from
  // compressedOffset, so we need to count down from the highest lane in the
  // warp
  auto prefix = __popc(vote & getLaneMaskGe());

  if (read) {
    // auto v = in[compressedOffset - prefix];
    auto v = in[-prefix];
    state = (state << kANSEncodedBits) + ANSStateT(v);
  }

  // how many values we actually read from the compressed input
  outNumRead = __popc(vote);
}

template <int ProbBits>
__device__ void decodeOnePartialWarp(
    bool valid,
    ANSStateT& state,

    // Start offset where this warp is reading from the
    // compressed input. As a variable number of lanes
    // wish to read from the compressed offset each
    // iteration, this offset upon calling is one after
    // the last offset, if any, this warp will be reading rom.
    uint32_t compressedOffset,

    const ANSEncodedT* __restrict__ in,

    // Shared memory LUTs
    const TableT* lookup,

    // Output: number of words read from compressed input
    uint32_t& outNumRead,

    // Output: decoded symbol for this iteration (only if valid)
    ANSDecodedT& outSym) {
  constexpr ANSStateT StateMask = (ANSStateT(1) << ProbBits) - ANSStateT(1);

  auto s_bar = state & StateMask;

  uint32_t sym;
  uint32_t pdf;
  uint32_t sMinusCdf;
  unpackDecodeLookup(lookup[s_bar], sym, pdf, sMinusCdf);

  if (valid) {
    outSym = sym;
    state = pdf * (state >> ProbBits) + ANSStateT(sMinusCdf);
  }

  // We only sometimes read a new encoded value
  bool read = valid && (state < kANSMinState);
  auto vote = __ballot_sync(0xffffffff, read);
  // We are reading in the same order as we wrote, except by decrementing from
  // compressedOffset, so we need to count down from the highest lane in the
  // warp
  auto prefix = __popc(vote & getLaneMaskGe());

  if (read) {
    // auto v = in[compressedOffset - prefix];
    auto v = in[-prefix];
    state = (state << kANSEncodedBits) + ANSStateT(v);
  }

  // how many values we actually read from the compressed input
  outNumRead = __popc(vote);
}

template <typename Writer, int ProbBits>
__device__ void ansDecodeWarpBlock(
    int laneId,
    ANSStateT state,
    uint32_t uncompressedWords,
    uint32_t compressedWords,
    const ANSEncodedT* __restrict__ in,
    Writer& writer,
    const TableT* __restrict__ table) {
  // The compressed input may not be a whole multiple of a warp.
  // In this case, only the lanes that cover the remainder could have read a
  // value in the input, and thus, only they can write a value in the output.
  // We handle this partial data first.
  uint32_t remainder = uncompressedWords % kWarpSize;

  // A fixed number of uncompressed elements are written each iteration
  int uncompressedOffset = uncompressedWords - remainder;

  // A variable number of compressed elements are read each iteration
  uint32_t compressedOffset = compressedWords;

  in += compressedOffset;

  // Partial warp handling the end of the data
  if (remainder) {
    bool valid = laneId < remainder;

    uint32_t numCompressedRead;
    ANSDecodedT sym;

    decodeOnePartialWarp<ProbBits>(
        valid, state, compressedOffset, in, table, numCompressedRead, sym);

    if (valid) {
      writer.write(uncompressedOffset + laneId, sym);
    }

    // compressedOffset -= numCompressedRead;
    in -= numCompressedRead;
  }

  // Full warp handling
  while (uncompressedOffset > 0) {
    uncompressedOffset -= kWarpSize;

    uint32_t numCompressedRead;
    ANSDecodedT sym;

    decodeOneWarp<ProbBits>(
        state, compressedOffset, in, table, numCompressedRead, sym);

    writer.write(uncompressedOffset + laneId, sym);

    // compressedOffset -= numCompressedRead;
    in -= numCompressedRead;
  }
}

template <typename Writer, int ProbBits, int BlockSize>
__device__ void ansDecodeWarpFullBlock(
    int laneId,
    ANSStateT state,
    uint32_t compressedWords,
    const ANSEncodedT* __restrict__ in,
    Writer& writer,
    const TableT* __restrict__ table) {
  // A variable number of compressed elements are read each iteration
  using VecT = ANSDecodedTx4;

  in += compressedWords;

  // 2: 252.16 us
  // 3: 246.62 us
  // 4: 254.91 us
  constexpr int kCacheLinesAhead = 3;

  for (int i = (BlockSize / sizeof(VecT)) - kWarpSize + laneId; i >= 0;
       i -= kWarpSize) {
    VecT symV;
    // Assuming no compression, we load 2 * sizeof(ANSEncodedT) *
    // kWarpSize = 128 bytes per iteration
    asm volatile("prefetch.global.L1 [%0];"
                 :
                 : "l"(in - (kCacheLinesAhead * 128) / sizeof(ANSEncodedT)));

    //    writer.preload(i + laneId);
    writer.preload(i);

#pragma unroll
    for (int j = sizeof(VecT) - 1; j >= 0; --j) {
      ANSDecodedT sym;
      uint32_t numCompressedRead;

      decodeOneWarp<ProbBits>(
          state, compressedWords, in, table, numCompressedRead, sym);

      symV.x[j] = sym;
      // compressedWords -= numCompressedRead;
      in -= numCompressedRead;
    }

    //    writer.writeVec(i + laneId, symV);
    writer.writeVec(i, symV);
  }
}

template <
    typename InProvider,
    typename OutProvider,
    int Threads,
    int ProbBits,
    int BlockSize>
__global__ __launch_bounds__(128) void ansDecodeKernel(
    InProvider inProvider,
    const TableT* __restrict__ table,
    OutProvider outProvider,
    uint8_t* __restrict__ outSuccess,
    uint32_t* __restrict__ outSize) {
  int tid = threadIdx.x;
  auto batch = blockIdx.y;

  // Interpret header as uint4
  auto headerIn = (const ANSCoalescedHeader*)inProvider.getBatchStart(batch);

  auto data0 = headerIn->data0;
  auto numBlocks = data0.x & 0xffffffU;
  auto totalUncompressedWords = data0.y;

  uint32_t symbolOffset;
  uint32_t probBits;
  uint32_t numSymbols;
  unpackSymbolInfo(data0.w, symbolOffset, probBits, numSymbols);
  // Is the data what we expect?
  assert(ProbBits == probBits);

  // Do we have enough space for the decompressed data?
  auto uncompressedBytes = totalUncompressedWords * sizeof(ANSDecodedT);
  bool success = outProvider.getBatchSize(batch) >= uncompressedBytes;

  if (blockIdx.x == 0 && tid == 0) {
    if (outSuccess) {
      outSuccess[batch] = success;
    }

    if (outSize) {
      outSize[batch] = uncompressedBytes;
    }
  }

  if (!success) {
    return;
  }

  // Initialize symbol, pdf, cdf tables
  constexpr int kBuckets = 1 << ProbBits;
  __shared__ TableT lookup[kBuckets];

  {
    uint4* lookup4 = (uint4*)lookup;
    const uint4* table4 = (const uint4*)(table + batch * (1 << ProbBits));

    static_assert(isEvenDivisor(kBuckets, Threads * 4), "");
    for (int j = 0;
         // loading by uint4 words
         j < kBuckets / (Threads * (sizeof(uint4) / sizeof(TableT)));
         ++j) {
      lookup4[j * Threads + tid] = table4[j * Threads + tid];
    }
  }

  __syncthreads();

  auto writer = outProvider.getWriter(batch);

  // warp id taking into account warps in the current block
  // do this so the compiler knows it is warp uniform
  int globalWarpId =
      __shfl_sync(0xffffffff, (blockIdx.x * blockDim.x + tid) / kWarpSize, 0);

  int warpsPerGrid = gridDim.x * Threads / kWarpSize;
  int laneId = getLaneId();

  for (int block = globalWarpId; block < numBlocks; block += warpsPerGrid) {
    // Load state
    ANSStateT state = headerIn->getWarpStates()[block].warpState[laneId];

    // Load per-block size data
    auto blockWords = headerIn->getBlockWords(numBlocks)[block];
    uint32_t uncompressedWords = (blockWords.x >> 16);
    uint32_t compressedWords = (blockWords.x & 0xffff);
    uint32_t blockCompressedWordStart = blockWords.y;

    // Get block addresses for encoded/decoded data
    auto blockDataIn =
        headerIn->getBlockDataStart(numBlocks) + blockCompressedWordStart;

    writer.setBlock(block);

    if (uncompressedWords == BlockSize) {
      ansDecodeWarpFullBlock<OutProvider::Writer, ProbBits, BlockSize>(
          laneId, state, compressedWords, blockDataIn, writer, lookup);
    } else {
      ansDecodeWarpBlock<OutProvider::Writer, ProbBits>(
          laneId,
          state,
          uncompressedWords,
          compressedWords,
          blockDataIn,
          writer,
          lookup);
    }
  }
}

template <typename BatchProvider, int Threads>
__global__ void ansDecodeTable(
    BatchProvider inProvider,
    uint32_t expectedProbBits,
    TableT* __restrict__ table) {
  int batch = blockIdx.x;
  int tid = threadIdx.x;
  int warpId = tid / kWarpSize;
  int laneId = getLaneId();

  table += batch * (1 << expectedProbBits);
  auto headerIn = (const ANSCoalescedHeader*)inProvider.getBatchStart(batch);

  // Is this an expected header?
  headerIn->checkMagic();

  uint32_t symbolOffset;
  uint32_t probBits;
  uint32_t numSymbols;
  unpackSymbolInfo(headerIn->symbolInfo(), symbolOffset, probBits, numSymbols);
  assert(probBits == expectedProbBits);

  if (numSymbols == 0) {
    // compressed empty array
    assert(headerIn->totalUncompressedWords() == 0);
    return;
  }

  // Skip to pdf table
  auto probs = headerIn->getSymbolProbs();

  static_assert(Threads >= kNumSymbols, "");
  uint32_t pdf = tid < numSymbols ? probs[tid] : 0;
  uint32_t cdf = 0;

  // Get the CDF from the PDF
  using BlockScan = cub::BlockScan<uint32_t, Threads>;
  __shared__ typename BlockScan::TempStorage tempStorage;

  uint32_t total = 0;
  // FIXME: don't use cub, we can write both the pdf and cdf to smem with a
  // single syncthreads
  BlockScan(tempStorage).ExclusiveSum(pdf, cdf, total);

  uint32_t totalProb = 1 << probBits;
  assert(totalProb == total); // should be a power of 2

  // Broadcast the pdf/cdf values
  __shared__ uint2 smemPdfCdf[kNumSymbols];

  if (tid < numSymbols) {
    smemPdfCdf[tid] = uint2{pdf, cdf};
  }

  __syncthreads();

  // Build the table for each pdf/cdf bucket
  constexpr int kWarpsPerBlock = Threads / kWarpSize;

  for (int i = warpId; i < numSymbols; i += kWarpsPerBlock) {
    auto v = smemPdfCdf[i];

    auto pdf = v.x;
    auto begin = v.y;
    auto end = (i + 1) < numSymbols ? (begin + pdf) : totalProb;

    for (int j = begin + laneId; j < end; j += kWarpSize) {
      table[j] = packDecodeLookup(
          i + symbolOffset, // symbol
          pdf, // bucket pdf
          j - begin); // within-bucket cdf
    }
  }
}

template <typename InProvider, typename OutProvider>
void ansDecodeBatch(
    StackDeviceMemory& res,
    int probBits,
    uint32_t numInBatch,
    const InProvider& inProvider,
    OutProvider& outProvider,
    uint8_t* outSuccess_dev,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  auto table_dev = res.alloc<TableT>(stream, numInBatch * (1 << probBits));

  // Build the rANS decoding table from the compression header
  {
    constexpr int kThreads = 512;
    ansDecodeTable<InProvider, kThreads><<<numInBatch, kThreads, 0, stream>>>(
        inProvider, probBits, table_dev.data());
  }

  // Perform decoding
  {
    // FIXME: We have no idea how large the decompression job is, as the
    // relevant information is on the device.
    // Just launch a grid that is sufficiently large enough to saturate the GPU;
    // blocks will exit if there isn't enough work, or will loop if there is
    // more work. We aim for a grid >4x larger than what the device can sustain,
    // to help cover up tail effects and unequal provisioning across the batch
#define RUN_DECODE(BITS)                                           \
  do {                                                             \
    constexpr int kThreads = 128;                                  \
    auto& props = getCurrentDeviceProperties();                    \
    int maxBlocksPerSM = 0;                                        \
    CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(     \
        &maxBlocksPerSM,                                           \
        ansDecodeKernel<                                           \
            InProvider,                                            \
            OutProvider,                                           \
            kThreads,                                              \
            BITS,                                                  \
            kDefaultBlockSize>,                                    \
        kThreads,                                                  \
        0));                                                       \
    uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount; \
    uint32_t perBatchGrid = divUp(maxGrid, numInBatch) * 4;        \
    auto grid = dim3(perBatchGrid, numInBatch);                    \
                                                                   \
    ansDecodeKernel<                                               \
        InProvider,                                                \
        OutProvider,                                               \
        kThreads,                                                  \
        BITS,                                                      \
        kDefaultBlockSize><<<grid, kThreads, 0, stream>>>(         \
        inProvider,                                                \
        table_dev.data(),                                          \
        outProvider,                                               \
        outSuccess_dev,                                            \
        outSize_dev);                                              \
  } while (false)

    switch (probBits) {
      case 9:
        RUN_DECODE(9);
        break;
      case 10:
        RUN_DECODE(10);
        break;
      case 11:
        RUN_DECODE(11);
        break;
      default:
        CHECK(false) << "unhandled pdf precision " << probBits;
    }

#undef RUN_DECODE
  }

  CUDA_TEST_ERROR();
}

} // namespace dietgpu
