/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <glog/logging.h>

namespace dietgpu {

constexpr uint32_t kGpuFloatHeaderMagic = 0x1234f00d;

// Header on our compressed floating point data
struct __align__(16) GpuFloatHeader {
  uint32_t magic;
  uint32_t size;
  uint32_t floatType;
  uint32_t unused;
};

static_assert(sizeof(GpuFloatHeader) == 16, "");

struct __align__(16) uint16x8 {
  uint16_t x[8];
};

struct __align__(8) uint16x4 {
  uint16_t x[4];
};

struct __align__(8) uint8x8 {
  uint8_t x[8];
};

inline size_t getWordSizeFromFloatType(FloatType ft) {
  switch (ft) {
    case FloatType::kFloat16:
    case FloatType::kBFloat16:
      return sizeof(uint16_t);
    // case FloatType::kFloat32:
    //   return sizeof(uint32_t);
    default:
      CHECK(ft == FloatType::kFloat16 || ft == FloatType::kBFloat16);
      return sizeof(uint16_t);
  }
}

} // namespace dietgpu
