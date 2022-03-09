/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dietgpu/ans/BatchProvider.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/PtxUtils.cuh"
#include "dietgpu/utils/StaticUtils.h"

#include <cmath>
#include <cub/cub.cuh>
#include <memory>

namespace dietgpu {

template <int Threads>
__device__ void histogramSingle(
    const ANSDecodedT* __restrict__ in,
    uint32_t size,
    uint32_t* __restrict__ out) {
  constexpr int kWarps = Threads / kWarpSize;
  static_assert(Threads == kNumSymbols, "");

  // +1 in order to force very common symbols that could overlap into different
  // banks?
  __shared__ uint32_t buckets[kWarps][kNumSymbols + 1];

  int warpId = threadIdx.x / kWarpSize;

#pragma unroll
  for (int i = 0; i < kWarps; ++i) {
    buckets[i][threadIdx.x] = 0;
  }

  __syncthreads();

  uint32_t* warpBucket = buckets[warpId];

  // If the size of batch is smaller than the increment for alignment, we only
  // handle the batch
  auto roundUp4 = min(size, getAlignmentRoundUp<sizeof(uint4)>(in));

  // The size of data that remains after alignment
  auto remaining = size - roundUp4;

  // The size of data (in uint4 words) that we can process with alignment
  uint32_t numU4 = divDown(remaining, sizeof(uint4));

  auto inAligned = in + roundUp4;
  auto inAligned4 = (const uint4*)inAligned;

  // Handle the non-aligned portion that we have to load as single words, if any
  if (blockIdx.x == 0 && threadIdx.x < roundUp4) {
    static_assert(sizeof(uint4) <= Threads, "");
    atomicAdd(&warpBucket[in[threadIdx.x]], 1);
  }

  // Handle the portion that is aligned and uint4 vectorizable
  // 37.60 us / 80.76% gmem / 51.29% smem for uint4 on A100
  for (uint32_t i = blockIdx.x * Threads + threadIdx.x; i < numU4;
       i += gridDim.x * Threads) {
    uint4 v = inAligned4[i];

    {
      uint32_t x = v.x;
      atomicAdd(&warpBucket[x & 0xff], 1);
      x >>= 8;
      atomicAdd(&warpBucket[x & 0xff], 1);
      x >>= 8;
      atomicAdd(&warpBucket[x & 0xff], 1);
      x >>= 8;
      atomicAdd(&warpBucket[x], 1);
    }

    {
      uint32_t y = v.y;
      atomicAdd(&warpBucket[y & 0xff], 1);
      y >>= 8;
      atomicAdd(&warpBucket[y & 0xff], 1);
      y >>= 8;
      atomicAdd(&warpBucket[y & 0xff], 1);
      y >>= 8;
      atomicAdd(&warpBucket[y], 1);
    }

    {
      uint32_t z = v.z;
      atomicAdd(&warpBucket[z & 0xff], 1);
      z >>= 8;
      atomicAdd(&warpBucket[z & 0xff], 1);
      z >>= 8;
      atomicAdd(&warpBucket[z & 0xff], 1);
      z >>= 8;
      atomicAdd(&warpBucket[z], 1);
    }

    {
      uint32_t w = v.w;
      atomicAdd(&warpBucket[w & 0xff], 1);
      w >>= 8;
      atomicAdd(&warpBucket[w & 0xff], 1);
      w >>= 8;
      atomicAdd(&warpBucket[w & 0xff], 1);
      w >>= 8;
      atomicAdd(&warpBucket[w], 1);
    }
  }

  if (blockIdx.x == 0) {
    // Handle the remainder portion that doesn't comprise full words
    int i = numU4 * sizeof(uint4) + threadIdx.x;
    if (i < remaining) {
      atomicAdd(&warpBucket[inAligned[i]], 1);
    }
  }

  __syncthreads();

  uint32_t sum = buckets[0][threadIdx.x];
#pragma unroll
  for (int j = 1; j < kWarps; ++j) {
    sum += buckets[j][threadIdx.x];
  }

  // The count for the thread's bucket could be 0
  if (sum) {
    atomicAdd(&out[threadIdx.x], sum);
  }
}

template <typename InProvider, int Threads>
__global__ void histogramBatch(InProvider in, uint32_t* out) {
  int batch = blockIdx.y;
  out += batch * kNumSymbols;

  histogramSingle<Threads>(
      (const ANSDecodedT*)in.getBatchStart(batch), in.getBatchSize(batch), out);
}

template <typename T>
struct Identity;

template <>
struct Identity<int> {
  static constexpr int kMin = std::numeric_limits<int>::max();
  static constexpr int kMax = std::numeric_limits<int>::min();
  static constexpr int kSum = 0;
};

template <int Threads>
__device__ inline void statsReduce(
    int warpId,
    int laneId,
    int valForMin,
    int valForMax,
    int valForSum,
    int& minOut,
    int& maxOut,
    int& sumOut) {
  static_assert(isEvenDivisor(Threads, kWarpSize), "");
  constexpr int kWarps = Threads / kWarpSize;

  auto allMin = warpReduceAllMin(valForMin);
  auto allMax = warpReduceAllMax(valForMax);
  auto allSum = warpReduceAllSum(valForSum);

  __shared__ int smemMin[kWarps];
  __shared__ int smemMax[kWarps];
  __shared__ int smemSum[kWarps];

  if (laneId == 0) {
    smemMin[warpId] = allMin;
    smemMax[warpId] = allMax;
    smemSum[warpId] = allSum;
  }
  __syncthreads();

  if (warpId == 0) {
    int v = laneId < kWarps ? smemMin[laneId] : Identity<int>::kMin;
    v = warpReduceAllMin(v);

    if (laneId == 0) {
      smemMin[0] = v;
    }
  } else if (warpId == 1) {
    int v = laneId < kWarps ? smemMax[laneId] : Identity<int>::kMax;
    v = warpReduceAllMax(v);

    if (laneId == 0) {
      smemMax[0] = v;
    }
  } else if (warpId == 2) {
    int v = laneId < kWarps ? smemSum[laneId] : Identity<int>::kSum;
    v = warpReduceAllSum(v);

    if (laneId == 0) {
      smemSum[0] = v;
    }
  }

  __syncthreads();

  // read broadcast values
  minOut = smemMin[0];
  maxOut = smemMax[0];
  sumOut = smemSum[0];
}

template <typename SizeProvider, int Threads>
__global__ void quantizeWeights(
    const uint32_t* __restrict__ counts,
    SizeProvider sizeProvider,
    int probBits,
    uint2* __restrict__ minAndNumSymbols,
    uint4* __restrict__ table) {
  static_assert(kNumSymbols <= Threads);

  // get to the right batch set of data
  int batch = blockIdx.x;
  minAndNumSymbols += batch;
  counts += batch * kNumSymbols;
  table += batch * kNumSymbols;

  uint32_t totalSize = sizeProvider.getBatchSize(batch);

  // There's nothing to do if the input array in the batch was of zero size
  if (totalSize == 0) {
    if (threadIdx.x == 0) {
      *minAndNumSymbols = uint2{0U, 0U};
    }

    return;
  }

  constexpr int kWarps = Threads / kWarpSize;
  uint32_t kProbWeight = 1 << probBits;
  int tid = threadIdx.x;
  int warpId = tid / kWarpSize;
  int laneId = getLaneId();

  // Load the current count and compute the min/max non-zero values
  uint32_t count = counts[tid];
  bool symPresent = count > 0;

  int minSym = symPresent ? tid : 0xffff;
  int maxSym = symPresent ? tid : -1;

  // Perform an approximate quantization using the above values
  uint32_t qProb = kProbWeight * ((float)count / (float)totalSize);

  // All weights for symbols present must be at least one
  qProb = (count > 0 && qProb == 0) ? 1 : qProb;

  int qProbSum;
  statsReduce<Threads>(
      warpId, laneId, minSym, maxSym, qProb, minSym, maxSym, qProbSum);

  int numSymbols = maxSym - minSym + 1;

  // In order to use radix sorting, and also in order to only sort a single
  // word, pack both the weight and index into a single integer
  uint32_t sortedPair = (qProb << 16) | tid /* symbol */;

  using Sort = cub::BlockRadixSort<uint32_t, Threads, 1>;
  __shared__ typename Sort::TempStorage smemSort;

  {
    uint32_t toSort[1];
    toSort[0] = sortedPair;
    Sort(smemSort).SortDescending(toSort);
    sortedPair = toSort[0];
  }

  // The (prob, symbol) pair that each thread is considered to
  // hold is the following:
  uint32_t tidSymbol = sortedPair & 0xffff;
  qProb = sortedPair >> 16;

  int diff = (int)kProbWeight - (int)qProbSum;

  if (diff > 0) {
    // add to largest values
    // FIXME: use div/mod to avoid iterations
    while (diff > 0) {
      int iterToApply = diff < numSymbols ? diff : numSymbols;

      if (tid < iterToApply) {
        qProb += 1;
      }

      diff -= iterToApply;
    }
  } else if (diff < 0) {
    diff = -diff;

    while (diff > 0) {
      // We need to determine the remaining number of >1 values
      // FIXME: clean up
      int qNumGt1s = warpReduceAllSum((int)(qProb > 1));
      __shared__ int smemSum2[kWarps];

      if (laneId == 0) {
        smemSum2[warpId] = qNumGt1s;
      }
      __syncthreads();

      if (warpId == 0) {
        qNumGt1s = laneId < kWarps ? smemSum2[laneId] : 0;
        qNumGt1s = warpReduceAllSum(qNumGt1s);
        if (laneId == 0) {
          smemSum2[0] = qNumGt1s;
        }
      }
      __syncthreads();

      qNumGt1s = smemSum2[0];

      // subtract from smallest >1 values
      // This should be the index of the first 1 value
      // FIXME: use div/mod to avoid iterations
      int iterToApply = diff < qNumGt1s ? diff : qNumGt1s;
      assert(iterToApply > 0);
      int startIndex = qNumGt1s - iterToApply;

      if (tid >= startIndex && tid < qNumGt1s) {
        qProb -= 1;
      }

      diff -= iterToApply;

      __syncthreads();
    }
  }

  // Recover the pre-sort order
  __shared__ uint32_t smemPdf[kNumSymbols];

  smemPdf[tidSymbol] = qProb;

  __syncthreads();

  uint32_t symPdf = smemPdf[tid];

  using Scan = cub::BlockScan<uint32_t, Threads>;
  __shared__ typename Scan::TempStorage smemScan;

  uint32_t symCdf = 0;
  Scan(smemScan).ExclusiveSum(symPdf, symCdf);

  // Compute divisor information (constant division via integer
  // multiplication + shift)
  uint32_t shift = 32 - __clz(symPdf - 1);

  constexpr uint64_t one = 1;
  uint64_t magic = ((one << 32) * ((one << shift) - symPdf)) / symPdf + 1;

  // magic should be able to fit
  table[tid] = uint4{symPdf, symCdf, (uint32_t)magic, shift};

  if (tid == 0) {
    *minAndNumSymbols = uint2{(uint32_t)minSym, (uint32_t)numSymbols};
  }
}

template <typename InProvider>
void ansHistogramBatch(
    uint32_t numInBatch,
    InProvider inProvider,
    // size numInBatch * kNumSymbols
    uint32_t* histogram_dev,
    cudaStream_t stream) {
  // 1. Compute symbol histogram
  // zero out buckets before proceeding, as we aggregate with atomic adds
  CUDA_VERIFY(cudaMemsetAsync(
      histogram_dev, 0, sizeof(uint32_t) * kNumSymbols * numInBatch, stream));

  {
    constexpr uint32_t kThreads = kNumSymbols;

    // What is the maximum number of blocks to saturate the GPU?
    int maxBlocks = 0;
    CUDA_VERIFY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocks, histogramBatch<InProvider, kThreads>, kThreads, 0));
    maxBlocks *= getCurrentDeviceProperties().multiProcessorCount;

    // The y block dimension will be for each batch element
    uint32_t xBlocks = divUp(maxBlocks, numInBatch);
    auto grid = dim3(xBlocks, numInBatch);

    histogramBatch<InProvider, kThreads>
        <<<grid, kThreads, 0, stream>>>(inProvider, histogram_dev);
  }
}

template <typename SizeProvider>
inline void ansCalcWeights(
    uint32_t numInBatch,
    int probBits,
    // we only use this for sizes (of each input batch member)
    SizeProvider sizeProvider,
    // size numInBatch * kNumSymbols
    const uint32_t* histogram_dev,
    // size numInBatch
    uint2* minAndNumSymbols_dev,
    // size numInBatch * kNumSymbols
    uint4* table_dev,
    cudaStream_t stream) {
  // Quantize weights and determine division factors
  quantizeWeights<SizeProvider, kNumSymbols>
      <<<numInBatch, kNumSymbols, 0, stream>>>(
          histogram_dev,
          sizeProvider,
          probBits,
          minAndNumSymbols_dev,
          table_dev);
}

} // namespace dietgpu
