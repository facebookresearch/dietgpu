/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <assert.h>
#include <cuda.h>
#include "dietgpu/utils/DeviceDefs.cuh"
#include "dietgpu/utils/StaticUtils.h"

namespace dietgpu {

using ANSStateT = uint32_t;
using ANSEncodedT = uint16_t;
using ANSDecodedT = uint8_t;

struct __align__(16) ANSDecodedTx16 {
  ANSDecodedT x[16];
};

struct __align__(8) ANSDecodedTx8 {
  ANSDecodedT x[8];
};

struct __align__(4) ANSDecodedTx4 {
  ANSDecodedT x[4];
};

constexpr uint32_t kNumSymbols = 1 << (sizeof(ANSDecodedT) * 8);
static_assert(kNumSymbols > 1, "");

// Default block size for compression (in bytes)
constexpr uint32_t kDefaultBlockSize = 4096;

// limit state to 2^31 - 1, so as to prevent addition overflow in the integer
// division via mul and shift by constants
constexpr int kANSStateBits = sizeof(ANSStateT) * 8 - 1;
constexpr int kANSEncodedBits = sizeof(ANSEncodedT) * 8; // out bits
constexpr ANSStateT kANSEncodedMask =
    (ANSStateT(1) << kANSEncodedBits) - ANSStateT(1);

constexpr ANSStateT kANSStartState = ANSStateT(1)
    << (kANSStateBits - kANSEncodedBits);
constexpr ANSStateT kANSMinState = ANSStateT(1)
    << (kANSStateBits - kANSEncodedBits);

constexpr uint32_t kWarpHeaderMagic = 0x1234f0f0;
constexpr uint32_t kCoalescedHeaderMagic = 0x5c;

// Each block of compressed data (either coalesced or uncoalesced) is aligned to
// this number of bytes and has a valid (if not all used) segment with this
// multiple of bytes
constexpr uint32_t kBlockAlignment = 16;

struct ANSWarpState {
  // The ANS state data for this warp
  ANSStateT warpState[kWarpSize];
};

inline __host__ __device__ uint32_t
packSymbolInfo(uint32_t symbolOffset, uint32_t probBits, uint32_t numSymbols) {
  uint32_t v = numSymbols;
  v <<= 8;
  v |= probBits;
  v <<= 8;
  v |= symbolOffset;

  return v;
}

inline __host__ __device__ void unpackSymbolInfo(
    uint32_t v,
    uint32_t& symbolOffset,
    uint32_t& probBits,
    uint32_t& numSymbols) {
  symbolOffset = (v & 0xffU);
  v >>= 8;
  probBits = (v & 0xffU);
  v >>= 8;
  numSymbols = v;
}

struct ANSCoalescedHeader {
  static __host__ __device__ uint32_t
  getCompressedOverhead(uint32_t numBlocks) {
    constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
        ? 1
        : kBlockAlignment / sizeof(uint2);

    return sizeof(ANSCoalescedHeader) +
        // probs
        sizeof(uint16_t) * kNumSymbols +
        // states
        sizeof(ANSWarpState) * numBlocks +
        // block words
        sizeof(uint2) * roundUp(numBlocks, kAlignment);
  }

  __host__ __device__ uint32_t getTotalCompressedSize() const {
    return getCompressedOverhead() +
        totalCompressedWords() * sizeof(ANSEncodedT);
  }

  __host__ __device__ uint32_t getCompressedOverhead() const {
    return getCompressedOverhead(numBlocks());
  }

  __host__ __device__ float getCompressionRatio() const {
    return (float)getTotalCompressedSize() / (float)totalUncompressedWords() *
        sizeof(ANSDecodedT);
  }

  __host__ __device__ uint32_t numBlocks() const {
    return (data0.x & 0xffffffU);
  }

  __host__ __device__ void setNumBlocks(uint32_t numBlocks) {
    assert(numBlocks <= 0xffffffU);
    data0.x = (kCoalescedHeaderMagic << 24) | numBlocks;
  }

  __host__ __device__ void checkMagic() const {
    assert((data0.x >> 24) == kCoalescedHeaderMagic);
  }

  __host__ __device__ uint32_t totalUncompressedWords() const {
    return data0.y;
  }

  __host__ __device__ void setUncompressedWords(uint32_t words) {
    data0.y = words;
  }

  __host__ __device__ uint32_t totalCompressedWords() const {
    return data0.z;
  }

  __host__ __device__ void setCompressedWords(uint32_t words) {
    data0.z = words;
  }

  __host__ __device__ uint32_t symbolInfo() const {
    return data0.w;
  }

  __host__ __device__ void setSymbolInfo(uint32_t info) {
    data0.w = info;
  }

  __device__ uint16_t* getSymbolProbs() {
    return (uint16_t*)(this + 1);
  }

  __device__ const uint16_t* getSymbolProbs() const {
    return (const uint16_t*)(this + 1);
  }

  __device__ ANSWarpState* getWarpStates() {
    return (ANSWarpState*)(getSymbolProbs() + kNumSymbols);
  }

  __device__ const ANSWarpState* getWarpStates() const {
    return (const ANSWarpState*)(getSymbolProbs() + kNumSymbols);
  }

  __device__ uint2* getBlockWords(uint32_t numBlocks) {
    // All of the ANSWarpStates are already kBlockAlignment aligned
    return (uint2*)(getWarpStates() + numBlocks);
  }

  __device__ const uint2* getBlockWords(uint32_t numBlocks) const {
    // All of the ANSWarpStates are already kBlockAlignment aligned
    return (const uint2*)(getWarpStates() + numBlocks);
  }

  __device__ ANSEncodedT* getBlockDataStart(uint32_t numBlocks) {
    constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
        ? 1
        : kBlockAlignment / sizeof(uint2);

    return (
        ANSEncodedT*)(getBlockWords(numBlocks) + roundUp(numBlocks, kAlignment));
  }

  __device__ const ANSEncodedT* getBlockDataStart(uint32_t numBlocks) const {
    constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
        ? 1
        : kBlockAlignment / sizeof(uint2);

    return (
        const ANSEncodedT*)(getBlockWords(numBlocks) + roundUp(numBlocks, kAlignment));
  }

  // x: (8: magic)(24: numBlocks)
  // y: totalUncompressedWords
  // z: totalCompressedWords
  // w: symbolInfo
  uint4 data0;

  // Fixed length array
  // uint16_t probs[kNumSymbols];

  // Variable length array:
  // ANSWarpState states[numBlocks];

  // Per-block information:
  // (uint16: uncompressedWords, uint16: compressedWords)
  // uint32: blockCompressedWordStart
  //
  // Variable length array:
  // uint2 blockWords[roundUp(numBlocks, kBlockAlignment / sizeof(uint2))];
};

static_assert(isEvenDivisor(sizeof(ANSCoalescedHeader), sizeof(uint4)), "");

} // namespace dietgpu
