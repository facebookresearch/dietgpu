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

struct __align__(16) uint32x4 {
  uint32_t x[4];
};

struct __align__(16) uint16x8 {
  uint16_t x[8];
};

struct __align__(8) uint16x4 {
  uint16_t x[4];
};

struct __align__(8) uint8x8 {
  uint8_t x[8];
};

struct __align__(4) uint8x4 {
  uint8_t x[4];
};

// Convert FloatType to word size/type
template <FloatType FT>
struct FloatTypeInfo;

template <>
struct FloatTypeInfo<FloatType::kFloat16> {
  using WordT = uint16_t;
  using CompT = uint8_t;
  using NonCompT = uint8_t;

  // How many bytes are not compressed?
  static constexpr size_t kNotCompressed = 1;

  // 16 byte vector type
  using VecT = uint16x8;
  using CompVecT = uint8x8;
  using NonCompVecT = uint8x8;

  using Vec4 = uint16x4;
  using CompVec4 = uint8x4;
  using NonCompVec4 = uint8x4;
};

template <>
struct FloatTypeInfo<FloatType::kBFloat16> {
  using WordT = uint16_t;
  using CompT = uint8_t;
  using NonCompT = uint8_t;

  // How many bytes are not compressed?
  static constexpr size_t kNotCompressed = 1;

  // 16 byte vector type
  using VecT = uint16x8;
  using CompVecT = uint8x8;
  using NonCompVecT = uint8x8;

  using Vec4 = uint16x4;
  using CompVec4 = uint8x4;
  using NonCompVec4 = uint8x4;
};

template <>
struct FloatTypeInfo<FloatType::kFloat32> {
  using WordT = uint32_t;
  using CompT = uint8_t;
  using NonCompT = uint32_t;

  // How many bytes are not compressed?
  // FIXME: pack to 3
  static constexpr size_t kNotCompressed = 4;

  // 16 byte vector type
  using VecT = uint32x4;
  using CompVecT = uint8x4;
  using NonCompVecT = uint32x4;

  using Vec4 = uint32x4;
  using CompVec4 = uint8x4;
  // FIXME
  using NonCompVec4 = uint32x4;
};

inline size_t getWordSizeFromFloatType(FloatType ft) {
  switch (ft) {
    case FloatType::kFloat16:
    case FloatType::kBFloat16:
      return sizeof(uint16_t);
    case FloatType::kFloat32:
      return sizeof(uint32_t);
    default:
      CHECK(false);
      return 0;
  }
}

} // namespace dietgpu
