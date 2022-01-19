/**
 * Copyright 2004-present Facebook. All Rights Reserved.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include "dietgpu/ans/GpuANSStatistics.cuh"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/utils/StackDeviceMemory.h"

using namespace dietgpu;

std::vector<uint8_t> generateSymbols(int num, float lambda = 20.0f) {
  std::random_device rd;
  std::mt19937 gen(10);
  std::exponential_distribution<float> dist(lambda);

  auto out = std::vector<uint8_t>(num);
  for (auto& v : out) {
    auto sample = std::min(dist(gen), 1.0f);

    v = sample * 256.0;
  }

  return out;
}

std::vector<uint32_t> histogram(
    const std::vector<uint8_t>& data,
    uint8_t* minSymbol = nullptr,
    uint8_t* maxSymbol = nullptr) {
  auto counts = std::vector<uint32_t>(256);

  int minSym = std::numeric_limits<int>::max();
  int maxSym = std::numeric_limits<int>::min();

  for (auto v : data) {
    minSym = std::min((int)v, minSym);
    maxSym = std::max((int)v, maxSym);
    counts[v]++;
  }

  EXPECT_GE(minSym, 0);
  EXPECT_LE(minSym, std::numeric_limits<uint8_t>::max());
  EXPECT_GE(maxSym, 0);
  EXPECT_LE(maxSym, std::numeric_limits<uint8_t>::max());
  EXPECT_LE(minSym, maxSym);

  if (minSymbol) {
    *minSymbol = minSym;
  }
  if (maxSymbol) {
    *maxSymbol = maxSym;
  }

  return counts;
}

TEST(ANSStatisticsTest, Histogram) {
  auto res = makeStackMemory();
  // run on a different stream to test stream assignment
  auto stream = CudaStream::makeNonBlocking();

  for (auto size :
       {1,
        2,
        11,
        32,
        55,
        1000,
        1001,
        1000000,
        1024 * 1024,
        1000001,
        12345677}) {
    int numInBatch = 3;

    auto data = std::vector<uint8_t>();

    auto histograms = std::vector<std::vector<uint32_t>>();

    int stride = 11;
    auto strideData = std::vector<uint8_t>(stride);

    for (int b = 0; b < numInBatch; ++b) {
      auto gen = generateSymbols(size, 20.0 + b * 2);
      histograms.push_back(histogram(gen));

      data.insert(data.end(), gen.begin(), gen.end());

      // Add some stride padding
      data.insert(data.end(), strideData.begin(), strideData.end());
    }

    auto data_dev = res.copyAlloc(stream, data);
    auto hist_dev = res.alloc<uint32_t>(stream, numInBatch * kNumSymbols);

    auto inProvider = BatchProviderStride(data_dev.data(), size + stride, size);

    ansHistogramBatch(numInBatch, inProvider, hist_dev.data(), stream);

    auto hist_host = hist_dev.copyToHost(stream);

    for (int b = 0; b < numInBatch; ++b) {
      for (int i = 0; i < kNumSymbols; ++i) {
        EXPECT_EQ(histograms[b][i], hist_host[b * kNumSymbols + i]);
      }
    }
  }
}

// TEST(ANSStatisticsTest, Statistics2) {
//   StandardGpuResources res;
//   auto stream = res.getDefaultStreamCurrentDevice();

//   for (auto size : {40000}) {
//     int numInBatch = 512;

//     std::vector<uint32_t> histograms;
//     auto data = std::vector<uint8_t>();

//     for (int b = 0; b < numInBatch; ++b) {
//       auto gen = generateSymbols(size, 20.0 + b * 2);
//       data.insert(data.end(), gen.begin(), gen.end());

//       auto hist = histogram(gen);
//       histograms.insert(histograms.end(), hist.begin(), hist.end());
//     }

//     auto data_dev = toDeviceNonTemporary(&res, data, stream);

//     auto histogram_dev = DeviceTensor<uint32_t, 1, true>(
//       &res, makeDevAlloc(AllocType::Other, stream),
//       {numInBatch * (int) kNumSymbols});

//     ansCalcHistogram(data_dev.data(),
//                      numInBatch,
//                      size,
//                      size,
//                      histogram_dev.data(),
//                      stream);

//     auto histograms_gpu = histogram_dev.copyToVector(stream);
//     EXPECT_EQ(histograms, histograms_gpu);
//   }
// }
