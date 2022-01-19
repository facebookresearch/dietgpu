/**
 * Copyright 2004-present Facebook. All Rights Reserved.
 */

#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSDecode.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

#include <glog/logging.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace dietgpu {

template <FloatType FT>
struct JoinFloat;

template <>
struct JoinFloat<FloatType::kFloat16> {
  static __device__ __forceinline__ uint16_t
  join(uint8_t comp, uint8_t nonComp) {
    uint16_t out = comp;
    out = (out << 8) | ((uint16_t)nonComp);

    return out;
  }
};

template <>
struct JoinFloat<FloatType::kBFloat16> {
  static __device__ __forceinline__ uint16_t
  join(uint8_t comp, uint8_t nonComp) {
    uint32_t lo = (uint32_t)comp * 256U + (uint32_t)nonComp;
    lo <<= 16;
    uint32_t hi = nonComp;

    uint32_t out;
    asm("shf.r.clamp.b32 %0, %1, %2, %3;"
        : "=r"(out)
        : "r"(lo), "r"(hi), "r"(1));
    return out >>= 16;
  }
};

template <FloatType FT, int Threads>
__device__ void joinFloatNonAligned(
    const uint8_t* __restrict__ compIn,
    const uint8_t* __restrict__ nonCompIn,
    uint32_t size,
    uint16_t* __restrict__ out) {
  for (uint32_t i = blockIdx.x * Threads + threadIdx.x; i < size;
       i += gridDim.x * Threads) {
    out[i] = JoinFloat<FT>::join(compIn[i], nonCompIn[i]);
  }
}

template <FloatType FT, int Threads>
__device__ void joinFloatAligned(
    const uint8_t* __restrict__ compIn,
    const uint8_t* __restrict__ nonCompIn,
    uint32_t size,
    uint16_t* __restrict__ out) {
  constexpr int kOuterUnroll = 2;
  constexpr int kInnerUnroll = sizeof(uint16x8) / sizeof(uint16_t);
  static_assert(kInnerUnroll == 8, "");

  const uint8x8* compIn8 = (const uint8x8*)compIn;
  const uint8x8* nonCompIn8 = (const uint8x8*)nonCompIn;
  uint16x8* out8 = (uint16x8*)out;

  // Each block handles Threads * kOuterUnroll * kInnerUnroll inputs/outputs at
  // a time, or Threads * kOuterUnroll 16-byte words at a time

  constexpr int kWordsPerBlock = Threads * kOuterUnroll;
  constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
  uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

  // Handle by block
  uint32_t startBlock = blockIdx.x * kWordsPerBlock;
  compIn8 += startBlock + threadIdx.x;
  nonCompIn8 += startBlock + threadIdx.x;
  out8 += startBlock + threadIdx.x;

  for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                compIn8 += gridDim.x * kWordsPerBlock,
                nonCompIn8 += gridDim.x * kWordsPerBlock,
                out8 += gridDim.x * kWordsPerBlock) {
    uint8x8 comp[kOuterUnroll];
    uint8x8 nonComp[kOuterUnroll];

#pragma unroll
    for (uint32_t i = 0; i < kOuterUnroll; ++i) {
      comp[i] = compIn8[i * Threads];
      nonComp[i] = nonCompIn8[i * Threads];
    }

    uint16x8 v[kOuterUnroll];

#pragma unroll
    for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
      for (int j = 0; j < kInnerUnroll; ++j) {
        v[i].x[j] = JoinFloat<FT>::join(comp[i].x[j], nonComp[i].x[j]);
      }
    }

#pragma unroll
    for (uint32_t i = 0; i < kOuterUnroll; ++i) {
      out8[i * Threads] = v[i];
    }
  }

  // Handle last (partial) block
  for (uint32_t i =
           fullBlocks * kFloatsPerBlock + blockIdx.x * Threads + threadIdx.x;
       i < size;
       i += blockDim.x) {
    out[i] = JoinFloat<FT>::join(compIn[i], nonCompIn[i]);
  }
}

template <
    typename InProviderComp,
    typename InProviderNonComp,
    typename OutProvider,
    FloatType FT,
    int Threads>
__global__ void joinFloat(
    InProviderComp inProviderComp,
    InProviderNonComp inProviderNonComp,
    OutProvider outProvider,
    uint8_t* __restrict__ outSuccess,
    uint32_t* __restrict__ outSize) {
  int batch = blockIdx.y;

  auto curCompIn = (const uint8_t*)inProviderComp.getBatchStart(batch);
  auto curNonCompIn = (const uint8_t*)inProviderNonComp.getBatchStart(batch);
  auto curOut = (uint16_t*)outProvider.getBatchStart(batch);

  // FIXME: test out capacity

  if (outSuccess && !outSuccess[batch]) {
    // ANS decompression failed, so nothing for us to do
    return;
  }

  // Get size as a header
  GpuFloatHeader h = *((const GpuFloatHeader*)curNonCompIn);
  assert(h.magic == kGpuFloatHeaderMagic);

  auto curSize = h.size;
  curNonCompIn += sizeof(GpuFloatHeader);

  if (outSize && (curSize != outSize[batch])) {
    // Reported size mismatch between ANS decompression and fp unpacking
    assert(false);
    return;
  }

  // curCompIn should always be aligned, as we decompress into temporary memory
  auto compUnalignedBytes = getAlignmentRoundUp<sizeof(uint4)>(curCompIn);
  auto nonCompUnalignedBytes = getAlignmentRoundUp<sizeof(uint4)>(curNonCompIn);
  auto outUnalignedBytes = getAlignmentRoundUp<sizeof(uint4)>(curOut);

  if (compUnalignedBytes || nonCompUnalignedBytes || outUnalignedBytes) {
    joinFloatNonAligned<FT, Threads>(curCompIn, curNonCompIn, curSize, curOut);
  } else {
    joinFloatAligned<FT, Threads>(curCompIn, curNonCompIn, curSize, curOut);
  }
}

template <typename InProvider>
struct FloatANSProvider {
  __host__ FloatANSProvider(InProvider& provider) : inProvider_(provider) {}

  __device__ void* getBatchStart(uint32_t batch) {
    uint8_t* p = (uint8_t*)inProvider_.getBatchStart(batch);

    GpuFloatHeader h = *((GpuFloatHeader*)p);
    assert(h.magic == kGpuFloatHeaderMagic);

    // This is where the ANS compressed data begins
    return (void*)(p + sizeof(GpuFloatHeader) + roundUp(h.size, sizeof(uint4)));
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    const uint8_t* p = (const uint8_t*)inProvider_.getBatchStart(batch);

    GpuFloatHeader h = *((const GpuFloatHeader*)p);
    assert(h.magic == kGpuFloatHeaderMagic);

    // This is where the ANS compressed data begins
    return p + sizeof(GpuFloatHeader) + roundUp(h.size, sizeof(uint4));
  }

  InProvider inProvider_;
};

template <int N>
struct FloatANSProviderInline {
  __host__ FloatANSProviderInline(int num, const void** in) {
    CHECK_LE(num, N);
    for (int i = 0; i < num; ++i) {
      in_[i] = in[i];
    }
  }

  __device__ void* getBatchStart(uint32_t batch) {
    uint8_t* p = (uint8_t*)in_[batch];

    GpuFloatHeader h = *((GpuFloatHeader*)p);
    assert(h.magic == kGpuFloatHeaderMagic);

    // This is where the ANS compressed data begins
    return (void*)(p + sizeof(GpuFloatHeader) + roundUp(h.size, sizeof(uint4)));
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    const uint8_t* p = (const uint8_t*)in_[batch];

    GpuFloatHeader h = *((const GpuFloatHeader*)p);
    assert(h.magic == kGpuFloatHeaderMagic);

    // This is where the ANS compressed data begins
    return p + sizeof(GpuFloatHeader) + roundUp(h.size, sizeof(uint4));
  }

  const void* in_[N];
};

template <FloatType FT, uint32_t BlockSize>
struct JoinFloatWriter {
  __host__ __device__ JoinFloatWriter(uint16_t* out, const uint8_t* nonComp)
      : out_(out),
        nonComp_(nonComp),
        outBlock_(nullptr),
        nonCompBlock_(nullptr) {}

  __host__ __device__ void setBlock(uint32_t block) {
    outBlock_ = out_ + block * BlockSize;
    nonCompBlock_ = nonComp_ + block * BlockSize;
  }

  __device__ void write(uint32_t offset, uint8_t sym) {
    auto nonComp = nonCompBlock_[offset];
    outBlock_[offset] = JoinFloat<FT>::join(sym, nonComp);
  }

  __device__ void preload(uint32_t offset) {
    // We can preload this before decompressing all of the ANS compressed data
    // to hide memory latency
    preload_ = ((uint32_t*)nonCompBlock_)[offset];
  }

  __device__ void writeVec(uint32_t offset, ANSDecodedTx4 symV) {
    uint16x4 outV;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      auto v = preload_ & 0xff;
      preload_ >>= 8;

      outV.x[i] = JoinFloat<FT>::join(symV.x[i], v);
    }

    ((uint16x4*)outBlock_)[offset] = outV;
  }

  uint32_t preload_;
  uint16_t* out_;
  const uint8_t* nonComp_;
  uint16_t* outBlock_;
  const uint8_t* nonCompBlock_;
};

template <
    typename InProvider,
    typename OutProvider,
    FloatType FT,
    uint32_t BlockSize>
struct FloatOutProvider {
  using Writer = JoinFloatWriter<FT, BlockSize>;

  __host__ FloatOutProvider(InProvider& inProvider, OutProvider& outProvider)
      : inProvider_(inProvider), outProvider_(outProvider) {}

  __device__ Writer getWriter(uint32_t batch) {
    return Writer(
        (uint16_t*)outProvider_.getBatchStart(batch),
        (const uint8_t*)inProvider_.getBatchStart(batch) +
            sizeof(GpuFloatHeader));
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return outProvider_.getBatchSize(batch);
  }

  InProvider inProvider_;
  OutProvider outProvider_;
};

template <int N, FloatType FT, uint32_t BlockSize>
struct FloatOutProviderInline {
  using Writer = JoinFloatWriter<FT, BlockSize>;

  __host__ FloatOutProviderInline(
      int num,
      const void** in,
      void** out,
      const uint32_t* outCapacity) {
    CHECK_LE(num, N);
    for (int i = 0; i < num; ++i) {
      in_[i] = in[i];
      out_[i] = out[i];
      outCapacity_[i] = outCapacity[i];
    }
  }

  __device__ Writer getWriter(uint32_t batch) {
    return Writer(
        (uint16_t*)out_[batch],
        (const uint8_t*)in_[batch] + sizeof(GpuFloatHeader));
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return outCapacity_[batch];
  }

  const void* in_[N];
  void* out_[N];
  uint32_t outCapacity_[N];
};

template <typename InProvider, typename OutProvider>
void floatDecompressDevice(
    StackDeviceMemory& res,
    const FloatDecompressConfig& config,
    uint32_t numInBatch,
    InProvider& inProvider,
    OutProvider& outProvider,
    uint32_t maxCapacity,
    uint8_t* outSuccess_dev,
    uint32_t* outSize_dev,
    cudaStream_t stream) {
  // We can perform decoding in a single pass if all input data is 16 byte
  // aligned
  if (config.is16ByteAligned) {
    //
    // Fused kernel: perform decompression in a single pass
    //
    auto inProviderANS = FloatANSProvider<InProvider>(inProvider);

    switch (config.floatType) {
      case kFloat16: {
        auto outProviderANS = FloatOutProvider<
            InProvider,
            OutProvider,
            FloatType::kFloat16,
            kDefaultBlockSize>(inProvider, outProvider);

        ansDecodeBatch(
            res,
            config.probBits,
            numInBatch,
            inProviderANS,
            outProviderANS,
            outSuccess_dev,
            outSize_dev,
            stream);
      } break;
      case kBFloat16: {
        auto outProviderANS = FloatOutProvider<
            InProvider,
            OutProvider,
            FloatType::kBFloat16,
            kDefaultBlockSize>(inProvider, outProvider);

        ansDecodeBatch(
            res,
            config.probBits,
            numInBatch,
            inProviderANS,
            outProviderANS,
            outSuccess_dev,
            outSize_dev,
            stream);
      } break;
      default:
        CHECK(false);
        break;
    }
  } else {
    //
    // Two pass kernel: decompress the ANS compressed data, then rejoin with
    // uncompressed data
    //

    // Temporary space for the decompressed exponents
    // We need to ensure 16 byte alignment for the decompressed data due to
    // vectorization
    uint32_t maxCapacityAligned = roundUp(maxCapacity, sizeof(uint4));

    auto exp_dev = res.alloc<uint8_t>(stream, numInBatch * maxCapacityAligned);

    using InProviderANS = FloatANSProvider<InProvider>;
    auto inProviderANS = InProviderANS(inProvider);

    using OutProviderANS = BatchProviderStride;
    auto outProviderANS =
        OutProviderANS(exp_dev.data(), maxCapacityAligned, maxCapacityAligned);

    ansDecodeBatch(
        res,
        config.probBits,
        numInBatch,
        inProviderANS,
        outProviderANS,
        outSuccess_dev,
        outSize_dev,
        stream);

    // Rejoin the rest of the float word
#define RUN_JOIN(FLOAT_TYPE)                                                 \
  do {                                                                       \
    constexpr int kThreads = 256;                                            \
    auto& props = getCurrentDeviceProperties();                              \
    int maxBlocksPerSM = 0;                                                  \
    CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(               \
        &maxBlocksPerSM,                                                     \
        joinFloat<                                                           \
            OutProviderANS,                                                  \
            InProvider,                                                      \
            OutProvider,                                                     \
            FLOAT_TYPE,                                                      \
            kThreads>,                                                       \
        kThreads,                                                            \
        0));                                                                 \
    uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount;           \
    uint32_t perBatchGrid = divUp(maxGrid, numInBatch);                      \
    if ((perBatchGrid * numInBatch > maxGrid) && perBatchGrid > 1) {         \
      perBatchGrid -= 1;                                                     \
    }                                                                        \
    auto grid = dim3(perBatchGrid, numInBatch);                              \
                                                                             \
    joinFloat<OutProviderANS, InProvider, OutProvider, FLOAT_TYPE, kThreads> \
        <<<grid, kThreads, 0, stream>>>(                                     \
            outProviderANS,                                                  \
            inProvider,                                                      \
            outProvider,                                                     \
            outSuccess_dev,                                                  \
            outSize_dev);                                                    \
  } while (false)

    switch (config.floatType) {
      case kFloat16:
        RUN_JOIN(kFloat16);
        break;
      case kBFloat16:
        RUN_JOIN(kBFloat16);
        break;
      default:
        assert(false);
        break;
    }

#undef RUN_JOIN
  }

  CUDA_TEST_ERROR();
}

} // namespace dietgpu
