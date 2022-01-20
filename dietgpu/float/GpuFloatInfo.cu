/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/float/GpuFloatUtils.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

namespace dietgpu {

__global__ void floatGetCompressedInfo(
    const void** in,
    uint32_t numInBatch,
    uint32_t* outSizes,
    uint32_t* outTypes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInBatch) {
    auto header = *(GpuFloatHeader*)in[idx];

    uint32_t size = 0;
    uint32_t type = FloatType::kUndefined;

    assert(header.magic == kGpuFloatHeaderMagic);
    size = header.size;
    type = header.floatType;

    if (outSizes) {
      outSizes[idx] = size;
    }
    if (outTypes) {
      outTypes[idx] = type;
    }
  }
}

void floatGetCompressedInfo(
    StackDeviceMemory& res,
    const void** in,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outTypes_dev,
    cudaStream_t stream) {
  if (!outSizes_dev && !outTypes_dev) {
    return;
  }

  auto in_dev = res.copyAlloc<const void*>(stream, in, numInBatch);

  floatGetCompressedInfoDevice(
      res, in_dev.data(), numInBatch, outSizes_dev, outTypes_dev, stream);

  CUDA_TEST_ERROR();
}

void floatGetCompressedInfoDevice(
    StackDeviceMemory& res,
    const void** in_dev,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    uint32_t* outTypes_dev,
    cudaStream_t stream) {
  if (!outSizes_dev && !outTypes_dev) {
    return;
  }

  auto block = 128;
  auto grid = divUp(numInBatch, block);

  floatGetCompressedInfo<<<grid, block, 0, stream>>>(
      in_dev, numInBatch, outSizes_dev, outTypes_dev);

  CUDA_TEST_ERROR();
}

} // namespace dietgpu
