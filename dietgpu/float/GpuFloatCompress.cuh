/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/ans/GpuANSEncode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/PtxUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

#include <glog/logging.h>
#include <cmath>
#include <memory>
#include <vector>

namespace dietgpu {

template <FloatType FT>
struct SplitFloat;

template <>
struct SplitFloat<FloatType::kFloat16> {
  static __device__ __forceinline__ void
  split(uint16_t in, uint8_t& comp, uint8_t& nonComp) {
    nonComp = in & 0xff;
    // don't bother extracting the specific exponent
    comp = in >> 8;
  }
};

template <>
struct SplitFloat<FloatType::kBFloat16> {
  static __device__ __forceinline__ void
  split(uint16_t in, uint8_t& comp, uint8_t& nonComp) {
    uint32_t v = (uint32_t)in * 65536U + (uint32_t)in;

    v = rotateLeft(v, 1);
    nonComp = v & 0xff;
    comp = v >> 24;
  }
};

template <FloatType FT, int Threads>
__device__ void splitFloatNonAligned(
    const uint16_t* in,
    uint32_t size,
    uint8_t* compOut,
    uint8_t* nonCompOut,
    uint32_t* warpHistogram) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += gridDim.x * blockDim.x) {
    uint8_t comp;
    uint8_t nonComp;
    SplitFloat<FT>::split(in[i], comp, nonComp);

    atomicAdd(&warpHistogram[comp], 1);

    compOut[i] = comp;
    nonCompOut[i] = nonComp;
  }
}

template <FloatType FT, int Threads>
__device__ void splitFloatAligned8(
    const uint16_t* in,
    uint32_t size,
    uint8_t* compOut,
    uint8_t* nonCompOut,
    uint32_t* warpHistogram) {
  constexpr int kOuterUnroll = 2;
  constexpr int kInnerUnroll = sizeof(uint16x8) / sizeof(uint16_t);
  static_assert(kInnerUnroll == 8, "");

  const uint16x8* in8 = (const uint16x8*)in;
  uint8x8* compOut8 = (uint8x8*)compOut;
  uint8x8* nonCompOut8 = (uint8x8*)nonCompOut;

  // Each block handles Threads * kOuterUnroll * kInnerUnroll inputs/outputs at
  // a time, or Threads * kOuterUnroll 16-byte words at a time

  constexpr int kWordsPerBlock = Threads * kOuterUnroll;
  constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
  uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

  // Handle by block
  uint32_t startBlock = blockIdx.x * kWordsPerBlock;
  in8 += startBlock + threadIdx.x;
  compOut8 += startBlock + threadIdx.x;
  nonCompOut8 += startBlock + threadIdx.x;

  for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                in8 += gridDim.x * kWordsPerBlock,
                compOut8 += gridDim.x * kWordsPerBlock,
                nonCompOut8 += gridDim.x * kWordsPerBlock) {
    uint16x8 v[kOuterUnroll];

#pragma unroll
    for (uint32_t i = 0; i < kOuterUnroll; ++i) {
      v[i] = in8[i * Threads];
    }

    uint8x8 compV[kOuterUnroll];
    uint8x8 nonCompV[kOuterUnroll];

#pragma unroll
    for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
      for (int j = 0; j < kInnerUnroll; ++j) {
        uint8_t comp;
        uint8_t nonComp;
        SplitFloat<FT>::split(v[i].x[j], comp, nonComp);

        atomicAdd(&warpHistogram[comp], 1);

        compV[i].x[j] = comp;
        nonCompV[i].x[j] = nonComp;
      }
    }

#pragma unroll
    for (uint32_t i = 0; i < kOuterUnroll; ++i) {
      compOut8[i * Threads] = compV[i];
      nonCompOut8[i * Threads] = nonCompV[i];
    }
  }

  // Handle last (partial) block
  for (uint32_t i =
           fullBlocks * kFloatsPerBlock + blockIdx.x * Threads + threadIdx.x;
       i < size;
       i += blockDim.x) {
    uint8_t comp;
    uint8_t nonComp;
    SplitFloat<FT>::split(in[i], comp, nonComp);

    atomicAdd(&warpHistogram[comp], 1);

    compOut[i] = comp;
    nonCompOut[i] = nonComp;
  }
}

template <
    typename InProvider,
    typename NonCompProvider,
    FloatType FT,
    int Threads>
__global__ void splitFloat(
    InProvider inProvider,
    uint8_t* __restrict__ compOut,
    uint32_t compOutStride,
    NonCompProvider nonCompProvider,
    uint32_t* __restrict__ histogramOut) {
  constexpr int kWarps = Threads / kWarpSize;
  static_assert(Threads == kNumSymbols, "");

  int batch = blockIdx.y;
  int warpId = threadIdx.x / kWarpSize;

  histogramOut += batch * kNumSymbols;

  // +1 in order to force very common symbols that could overlap into different
  // banks between different warps
  __shared__ uint32_t histogram[kWarps][kNumSymbols + 1];

#pragma unroll
  for (int i = 0; i < kWarps; ++i) {
    histogram[i][threadIdx.x] = 0;
  }

  __syncthreads();

  uint32_t* warpHistogram = histogram[warpId];

  auto curIn = (const uint16_t*)inProvider.getBatchStart(batch);
  auto curNonCompOut = (uint8_t*)nonCompProvider.getBatchStart(batch);
  auto curCompOut = compOut + compOutStride * batch;
  auto curSize = inProvider.getBatchSize(batch);

  // Write size as a header
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    GpuFloatHeader h;
    h.magic = kGpuFloatHeaderMagic;
    h.size = curSize;
    h.floatType = FT;
    *((GpuFloatHeader*)curNonCompOut) = h;
  }

  curNonCompOut += sizeof(GpuFloatHeader);

  // How many bytes are before the point where we are 16 byte aligned?
  auto nonAlignedBytes = getAlignmentRoundUp<sizeof(uint16x8)>(curIn);

  if (nonAlignedBytes > 0) {
    splitFloatNonAligned<FT, Threads>(
        curIn, curSize, curCompOut, curNonCompOut, warpHistogram);
  } else {
    splitFloatAligned8<FT, Threads>(
        curIn, curSize, curCompOut, curNonCompOut, warpHistogram);
  }

  // Accumulate warp histogram data and write into the gmem histogram
  __syncthreads();

  uint32_t sum = histogram[0][threadIdx.x];
#pragma unroll
  for (int j = 1; j < kWarps; ++j) {
    sum += histogram[j][threadIdx.x];
  }

  // The count for the thread's bucket could be 0
  if (sum) {
    atomicAdd(&histogramOut[threadIdx.x], sum);
  }
}

// Update the final byte counts for the batch to take into account the
// uncompressed and compressed portions
template <typename InProvider>
__global__ void
incOutputSizes(InProvider inProvider, uint32_t* outSize, uint32_t numInBatch) {
  uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < numInBatch) {
    outSize[batch] += sizeof(GpuFloatHeader) +
        roundUp(inProvider.getBatchSize(batch), sizeof(uint4));
  }
}

// Provides the input data to ANS compression
template <typename SizeProvider>
struct FloatANSInProvider {
  using Writer = BatchWriter;

  __host__
  FloatANSInProvider(void* ptr_dev, uint32_t stride, SizeProvider& sizeProvider)
      : ptr_dev_(ptr_dev), stride_(stride), sizeProvider_(sizeProvider) {}

  __device__ void* getBatchStart(uint32_t batch) {
    return (uint8_t*)ptr_dev_ + batch * stride_;
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    return (uint8_t*)ptr_dev_ + batch * stride_;
  }

  __device__ BatchWriter getWriter(uint32_t batch) {
    return BatchWriter(getBatchStart(batch));
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return sizeProvider_.getBatchSize(batch);
  }

  void* ptr_dev_;
  uint32_t stride_;
  SizeProvider sizeProvider_;
};

// Provides the output data to ANS compression
template <typename OutProvider, typename SizeProvider>
struct FloatANSOutProvider {
  using Writer = BatchWriter;

  __host__ FloatANSOutProvider(
      OutProvider& outProvider,
      SizeProvider& sizeProvider)
      : outProvider_(outProvider), sizeProvider_(sizeProvider) {}

  __device__ void* getBatchStart(uint32_t batch) {
    uint8_t* p = (uint8_t*)outProvider_.getBatchStart(batch);

    // Increment the pointer to past the floating point data
    assert(((GpuFloatHeader*)p)->magic == kGpuFloatHeaderMagic);
    return p + sizeof(GpuFloatHeader) +
        roundUp(sizeProvider_.getBatchSize(batch), sizeof(uint4));
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    const uint8_t* p = (const uint8_t*)outProvider_.getBatchStart(batch);

    // Increment the pointer to past the floating point data
    assert(((GpuFloatHeader*)p)->magic == kGpuFloatHeaderMagic);
    return p + sizeof(GpuFloatHeader) +
        roundUp(sizeProvider_.getBatchSize(batch), sizeof(uint4));
  }

  __device__ BatchWriter getWriter(uint32_t batch) {
    return BatchWriter(getBatchStart(batch));
  }

  OutProvider outProvider_;
  SizeProvider sizeProvider_;
};

template <typename InProvider, typename OutProvider>
void floatCompressDevice(
    StackDeviceMemory& res,
    const FloatCompressConfig& config,
    uint32_t numInBatch,
    InProvider& inProvider,
    uint32_t maxSize,
    OutProvider& outProvider,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  auto maxUncompressedWords = maxSize / sizeof(ANSDecodedT);
  uint32_t maxNumCompressedBlocks =
      divUp(maxUncompressedWords, kDefaultBlockSize);

  // Temporary space for the extracted exponents; all rows must be 16 byte
  // aligned
  uint32_t compRowStride = roundUp(maxSize, sizeof(uint4));
  auto toComp_dev = res.alloc<uint8_t>(stream, numInBatch * compRowStride);

  // We calculate a histogram of the symbols to be compressed as part of
  // extracting the compressible symbol from the float
  auto histogram_dev = res.alloc<uint32_t>(stream, numInBatch * kNumSymbols);

  // zero out buckets before proceeding, as we aggregate with atomic adds
  CUDA_VERIFY(cudaMemsetAsync(
      histogram_dev.data(),
      0,
      sizeof(uint32_t) * numInBatch * kNumSymbols,
      stream));

#define RUN_SPLIT(FLOAT_TYPE)                                      \
  do {                                                             \
    constexpr int kBlock = 256;                                    \
    auto& props = getCurrentDeviceProperties();                    \
    int maxBlocksPerSM = 0;                                        \
    CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(     \
        &maxBlocksPerSM,                                           \
        splitFloat<InProvider, OutProvider, FLOAT_TYPE, kBlock>,   \
        kBlock,                                                    \
        0));                                                       \
    uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount; \
    uint32_t perBatchGrid = 4 * divUp(maxGrid, numInBatch);        \
    auto grid = dim3(perBatchGrid, numInBatch);                    \
                                                                   \
    splitFloat<InProvider, OutProvider, FLOAT_TYPE, kBlock>        \
        <<<grid, kBlock, 0, stream>>>(                             \
            inProvider,                                            \
            toComp_dev.data(),                                     \
            compRowStride,                                         \
            outProvider,                                           \
            histogram_dev.data());                                 \
  } while (false)

  switch (config.floatType) {
    case kFloat16:
      RUN_SPLIT(kFloat16);
      break;
    case kBFloat16:
      RUN_SPLIT(kBFloat16);
      break;
    default:
      assert(false);
      break;
  }

  // batch is strided, but sizes are original in a device array
  auto inProviderANS = FloatANSInProvider<InProvider>(
      toComp_dev.data(), compRowStride, inProvider);

  // We have written the non-compressed portions of the floats into the output,
  // along with a header that indicates how many floats there are.
  // For compression, we need to increment the address in which the compressed
  // outputs are written.
  auto outProviderANS =
      FloatANSOutProvider<OutProvider, InProvider>(outProvider, inProvider);

  ansEncodeBatchDevice(
      res,
      config.probBits,
      numInBatch,
      inProviderANS,
      histogram_dev.data(),
      maxSize,
      outProviderANS,
      outSize_dev,
      stream);

  // outSize as reported by ansEncode is just the ANS-encoded portion of the
  // data.
  // We need to increment the sizes by the uncompressed portion (header plus
  // uncompressed float data).
  {
    uint32_t block = 128;
    uint32_t grid = divUp(numInBatch, block);

    incOutputSizes<<<grid, block, 0, stream>>>(
        inProvider, outSize_dev, numInBatch);
  }

  CUDA_TEST_ERROR();
}

} // namespace dietgpu
