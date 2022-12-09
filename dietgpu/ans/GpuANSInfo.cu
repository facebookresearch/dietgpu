/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/ans/GpuANSUtils.cuh"
#include "dietgpu/utils/DeviceUtils.h"
#include "dietgpu/utils/StackDeviceMemory.h"
#include "dietgpu/utils/StaticUtils.h"

namespace dietgpu {

__global__ void
ansGetCompressedInfo(const void** in, uint32_t numInBatch, uint32_t* outSizes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numInBatch) {
    auto header = *(ANSCoalescedHeader*)in[idx];

    header.checkMagicAndVersion();
    outSizes[idx] = header.getTotalUncompressedWords();
  }
}

void ansGetCompressedInfo(
    StackDeviceMemory& res,
    const void** in,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    cudaStream_t stream) {
  if (!outSizes_dev) {
    return;
  }

  auto in_dev = res.copyAlloc<void*>(stream, (void**)in, numInBatch);

  ansGetCompressedInfoDevice(
      res, (const void**)in_dev.data(), numInBatch, outSizes_dev, stream);

  CUDA_TEST_ERROR();
}

void ansGetCompressedInfoDevice(
    StackDeviceMemory& res,
    const void** in_dev,
    uint32_t numInBatch,
    uint32_t* outSizes_dev,
    cudaStream_t stream) {
  if (!outSizes_dev) {
    return;
  }

  auto block = 128;
  auto grid = divUp(numInBatch, block);

  ansGetCompressedInfo<<<grid, block, 0, stream>>>(
      in_dev, numInBatch, outSizes_dev);

  CUDA_TEST_ERROR();
}

} // namespace dietgpu
