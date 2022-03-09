/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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
  static __device__ __forceinline__
      typename FloatTypeInfo<FloatType::kFloat16>::WordT
      join(
          typename FloatTypeInfo<FloatType::kFloat16>::CompT comp,
          typename FloatTypeInfo<FloatType::kFloat16>::NonCompT nonComp) {
    uint16_t out = comp;
    out = (out << 8) | ((uint16_t)nonComp);

    return out;
  }
};

template <>
struct JoinFloat<FloatType::kBFloat16> {
  static __device__ __forceinline__
      typename FloatTypeInfo<FloatType::kBFloat16>::WordT
      join(
          typename FloatTypeInfo<FloatType::kBFloat16>::CompT comp,
          typename FloatTypeInfo<FloatType::kBFloat16>::NonCompT nonComp) {
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

template <>
struct JoinFloat<FloatType::kFloat32> {
  static __device__ __forceinline__
      typename FloatTypeInfo<FloatType::kFloat32>::WordT
      join(
          typename FloatTypeInfo<FloatType::kFloat32>::CompT comp,
          typename FloatTypeInfo<FloatType::kFloat32>::NonCompT nonComp) {
    uint32_t v = (uint32_t(comp) * 16777216U) + uint32_t(nonComp);
    return rotateRight(v, 1);
  }
};

template <FloatType FT, int Threads>
__device__ void joinFloatNonAligned(
    const typename FloatTypeInfo<FT>::CompT* __restrict__ compIn,
    const typename FloatTypeInfo<FT>::NonCompT* __restrict__ nonCompIn,
    uint32_t size,
    typename FloatTypeInfo<FT>::WordT* __restrict__ out) {
  for (uint32_t i = blockIdx.x * Threads + threadIdx.x; i < size;
       i += gridDim.x * Threads) {
    out[i] = JoinFloat<FT>::join(compIn[i], nonCompIn[i]);
  }
}

template <FloatType FT, int Threads>
__device__ void joinFloatAligned(
    const typename FloatTypeInfo<FT>::CompT* __restrict__ compIn,
    const typename FloatTypeInfo<FT>::NonCompT* __restrict__ nonCompIn,
    uint32_t size,
    typename FloatTypeInfo<FT>::WordT* __restrict__ out) {
  using WordT = typename FloatTypeInfo<FT>::WordT;
  using CompT = typename FloatTypeInfo<FT>::CompT;
  using NonCompT = typename FloatTypeInfo<FT>::NonCompT;
  using VecT = typename FloatTypeInfo<FT>::VecT;
  using CompVecT = typename FloatTypeInfo<FT>::CompVecT;
  using NonCompVecT = typename FloatTypeInfo<FT>::NonCompVecT;

  constexpr int kOuterUnroll = 2;
  constexpr int kInnerUnroll = sizeof(VecT) / sizeof(WordT);

  const CompVecT* compInV = (const CompVecT*)compIn;
  const NonCompVecT* nonCompInV = (const NonCompVecT*)nonCompIn;
  VecT* outV = (VecT*)out;

  // Each block handles Threads * kOuterUnroll * kInnerUnroll inputs/outputs at
  // a time, or Threads * kOuterUnroll 16-byte words at a time

  constexpr int kWordsPerBlock = Threads * kOuterUnroll;
  constexpr int kFloatsPerBlock = kWordsPerBlock * kInnerUnroll;
  uint32_t fullBlocks = divDown(size, kFloatsPerBlock);

  // Handle by block
  uint32_t startBlock = blockIdx.x * kWordsPerBlock;
  compInV += startBlock + threadIdx.x;
  nonCompInV += startBlock + threadIdx.x;
  outV += startBlock + threadIdx.x;

  for (uint32_t b = blockIdx.x; b < fullBlocks; b += gridDim.x,
                compInV += gridDim.x * kWordsPerBlock,
                nonCompInV += gridDim.x * kWordsPerBlock,
                outV += gridDim.x * kWordsPerBlock) {
    CompVecT comp[kOuterUnroll];
    NonCompVecT nonComp[kOuterUnroll];

#pragma unroll
    for (uint32_t i = 0; i < kOuterUnroll; ++i) {
      comp[i] = compInV[i * Threads];
      nonComp[i] = nonCompInV[i * Threads];
    }

    VecT v[kOuterUnroll];

#pragma unroll
    for (uint32_t i = 0; i < kOuterUnroll; ++i) {
#pragma unroll
      for (int j = 0; j < kInnerUnroll; ++j) {
        v[i].x[j] = JoinFloat<FT>::join(comp[i].x[j], nonComp[i].x[j]);
      }
    }

#pragma unroll
    for (uint32_t i = 0; i < kOuterUnroll; ++i) {
      outV[i * Threads] = v[i];
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
  using WordT = typename FloatTypeInfo<FT>::WordT;
  using CompT = typename FloatTypeInfo<FT>::CompT;
  using NonCompT = typename FloatTypeInfo<FT>::NonCompT;

  int batch = blockIdx.y;

  auto curCompIn = (const CompT*)inProviderComp.getBatchStart(batch);
  auto curHeaderIn =
      (const GpuFloatHeader*)inProviderNonComp.getBatchStart(batch);
  auto curOut = (WordT*)outProvider.getBatchStart(batch);

  // FIXME: test out capacity

  if (outSuccess && !outSuccess[batch]) {
    // ANS decompression failed, so nothing for us to do
    return;
  }

  // Get size as a header
  GpuFloatHeader h = *curHeaderIn;
  assert(h.magic == kGpuFloatHeaderMagic);

  auto curSize = h.size;

  if (outSize && (curSize != outSize[batch])) {
    // Reported size mismatch between ANS decompression and fp unpacking
    assert(false);
    return;
  }

  auto curNonCompIn = (const NonCompT*)(curHeaderIn + 1);

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

template <FloatType FT, typename InProvider>
struct FloatANSProvider {
  using FTI = FloatTypeInfo<FT>;

  __host__ FloatANSProvider(InProvider& provider) : inProvider_(provider) {}

  __device__ void* getBatchStart(uint32_t batch) {
    uint8_t* p = (uint8_t*)inProvider_.getBatchStart(batch);

    GpuFloatHeader h = *((GpuFloatHeader*)p);
    assert(h.magic == kGpuFloatHeaderMagic);

    // Increment the pointer to past the floating point data
    return (
        void*)(p + sizeof(GpuFloatHeader) + roundUp(FTI::kNotCompressed * h.size, sizeof(uint4)));
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    const uint8_t* p = (const uint8_t*)inProvider_.getBatchStart(batch);

    GpuFloatHeader h = *((const GpuFloatHeader*)p);
    assert(h.magic == kGpuFloatHeaderMagic);

    // Increment the pointer to past the floating point data
    return p + sizeof(GpuFloatHeader) +
        roundUp(FTI::kNotCompressed * h.size, sizeof(uint4));
  }

  InProvider inProvider_;
};

template <FloatType FT, int N>
struct FloatANSProviderInline {
  using FTI = FloatTypeInfo<FT>;

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

    // Increment the pointer to past the floating point data
    return p + sizeof(GpuFloatHeader) +
        roundUp(FTI::kNotCompressed * h.size, sizeof(uint4));
  }

  __device__ const void* getBatchStart(uint32_t batch) const {
    const uint8_t* p = (const uint8_t*)in_[batch];

    GpuFloatHeader h = *((const GpuFloatHeader*)p);
    assert(h.magic == kGpuFloatHeaderMagic);

    // Increment the pointer to past the floating point data
    return p + sizeof(GpuFloatHeader) +
        roundUp(FTI::kNotCompressed * h.size, sizeof(uint4));
  }

  const void* in_[N];
};

template <FloatType FT, uint32_t BlockSize>
struct JoinFloatWriter {
  using FTI = FloatTypeInfo<FT>;

  __host__ __device__ JoinFloatWriter(
      typename FTI::WordT* out,
      const typename FTI::NonCompT* nonComp)
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

  // The preload is an offset of a NonCompVec4
  __device__ void preload(uint32_t offset) {
    // We can preload this before decompressing all of the ANS compressed data
    // to hide memory latency
    preload_ = ((typename FTI::NonCompVec4*)nonCompBlock_)[offset];
  }

  __device__ void writeVec(uint32_t offset, ANSDecodedTx4 symV) {
    typename FTI::Vec4 outV;
#pragma unroll
    // We always receive 4 decoded values each iteration
    // FIXME: this is hacky
    for (int i = 0; i < 4; ++i) {
      outV.x[i] = JoinFloat<FT>::join(symV.x[i], preload_.x[i]);
    }

    ((typename FTI::Vec4*)outBlock_)[offset] = outV;
  }

  typename FTI::NonCompVec4 preload_;
  typename FTI::WordT* out_;
  const typename FTI::NonCompT* nonComp_;
  typename FTI::WordT* outBlock_;
  const typename FTI::NonCompT* nonCompBlock_;
};

template <
    typename InProvider,
    typename OutProvider,
    FloatType FT,
    uint32_t BlockSize>
struct FloatOutProvider {
  using Writer = JoinFloatWriter<FT, BlockSize>;
  using FTI = FloatTypeInfo<FT>;

  __host__ FloatOutProvider(InProvider& inProvider, OutProvider& outProvider)
      : inProvider_(inProvider), outProvider_(outProvider) {}

  __device__ Writer getWriter(uint32_t batch) {
    return Writer(
        (typename FTI::WordT*)outProvider_.getBatchStart(batch),
        (const typename FTI::NonCompT*)
        // advance past the header
        (((GpuFloatHeader*)inProvider_.getBatchStart(batch)) + 1));
  }

  __device__ uint32_t getBatchSize(uint32_t batch) {
    return outProvider_.getBatchSize(batch);
  }

  InProvider inProvider_;
  OutProvider outProvider_;
};

template <int N, FloatType FT, uint32_t BlockSize>
struct FloatOutProviderInline {
  using FTI = FloatTypeInfo<FT>;
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
        (typename FTI::WordT*)out_[batch],
        (const typename FTI::
             NonCompT*)((const uint8_t*)in_[batch] + sizeof(GpuFloatHeader)));
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

#define RUN_FUSED(FT)                                                     \
  do {                                                                    \
    auto inProviderANS = FloatANSProvider<FT, InProvider>(inProvider);    \
    auto outProviderANS =                                                 \
        FloatOutProvider<InProvider, OutProvider, FT, kDefaultBlockSize>( \
            inProvider, outProvider);                                     \
                                                                          \
    ansDecodeBatch(                                                       \
        res,                                                              \
        config.probBits,                                                  \
        numInBatch,                                                       \
        inProviderANS,                                                    \
        outProviderANS,                                                   \
        outSuccess_dev,                                                   \
        outSize_dev,                                                      \
        stream);                                                          \
  } while (false)

    switch (config.floatType) {
      case kFloat16:
        RUN_FUSED(FloatType::kFloat16);
        break;
      case kBFloat16:
        RUN_FUSED(FloatType::kBFloat16);
        break;
      case kFloat32:
        RUN_FUSED(FloatType::kFloat32);
        break;
      default:
        CHECK(false);
        break;
    }

#undef RUN_FUSED
  }

  else {
    //
    // Two pass kernel: decompress the ANS compressed data, then rejoin with
    // uncompressed data
    //

    // Temporary space for the decompressed exponents
    // We need to ensure 16 byte alignment for the decompressed data due to
    // vectorization
    uint32_t maxCapacityAligned = roundUp(maxCapacity, sizeof(uint4));

    auto exp_dev = res.alloc<uint8_t>(stream, numInBatch * maxCapacityAligned);

#define RUN_DECODE(FT)                                                    \
  do {                                                                    \
    using InProviderANS = FloatANSProvider<FT, InProvider>;               \
    auto inProviderANS = InProviderANS(inProvider);                       \
                                                                          \
    using OutProviderANS = BatchProviderStride;                           \
    auto outProviderANS = OutProviderANS(                                 \
        exp_dev.data(), maxCapacityAligned, maxCapacityAligned);          \
                                                                          \
    ansDecodeBatch(                                                       \
        res,                                                              \
        config.probBits,                                                  \
        numInBatch,                                                       \
        inProviderANS,                                                    \
        outProviderANS,                                                   \
        outSuccess_dev,                                                   \
        outSize_dev,                                                      \
        stream);                                                          \
                                                                          \
    constexpr int kThreads = 256;                                         \
    auto& props = getCurrentDeviceProperties();                           \
    int maxBlocksPerSM = 0;                                               \
    CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(            \
        &maxBlocksPerSM,                                                  \
        joinFloat<OutProviderANS, InProvider, OutProvider, FT, kThreads>, \
        kThreads,                                                         \
        0));                                                              \
    uint32_t maxGrid = maxBlocksPerSM * props.multiProcessorCount;        \
    uint32_t perBatchGrid = divUp(maxGrid, numInBatch);                   \
    if ((perBatchGrid * numInBatch > maxGrid) && perBatchGrid > 1) {      \
      perBatchGrid -= 1;                                                  \
    }                                                                     \
    auto grid = dim3(perBatchGrid, numInBatch);                           \
                                                                          \
    joinFloat<OutProviderANS, InProvider, OutProvider, FT, kThreads>      \
        <<<grid, kThreads, 0, stream>>>(                                  \
            outProviderANS,                                               \
            inProvider,                                                   \
            outProvider,                                                  \
            outSuccess_dev,                                               \
            outSize_dev);                                                 \
  } while (false)

    switch (config.floatType) {
      case kFloat16:
        RUN_DECODE(FloatType::kFloat16);
        break;
      case kBFloat16:
        RUN_DECODE(FloatType::kBFloat16);
        break;
      case kFloat32:
        RUN_DECODE(FloatType::kFloat32);
        break;
      default:
        CHECK(false);
        break;
    }

#undef RUN_DECODE
  }

  CUDA_TEST_ERROR();
}

} // namespace dietgpu
