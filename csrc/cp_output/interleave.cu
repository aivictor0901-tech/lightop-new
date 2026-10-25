// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace lightop::cp_output {
namespace {

constexpr int kCycleThreads = 64;
constexpr int kItemsPerThread = 2;
constexpr int kGenericThreads = 256;
constexpr int kGenericItems = 8;
constexpr int64_t kMaxBlocks = 65535;

#if defined(__HIP_PLATFORM_AMD__)
using Vector = uint32_t __attribute__((ext_vector_type(4)));
#else
using Vector = uint4;
#endif
static_assert(sizeof(Vector) == 16);

// One (cycle, column tile) owns disjoint bytes for the entire permutation.
// Load all rows into registers before writing; no global scratch or inter-block
// barrier is needed. Power-of-two layouts have a small known cycle period.
template <int Period>
__global__ __launch_bounds__(kCycleThreads) void short_cycles(
    Vector* data, const int64_t* leaders, int64_t cycles, int64_t row_vectors,
    int cp_shift, int local_shift) {
  constexpr int64_t kTile = kCycleThreads * kItemsPerThread;
  const int64_t tiles = (row_vectors + kTile - 1) / kTile;
  for (int64_t task = blockIdx.x; task < cycles * tiles; task += gridDim.x) {
    const int64_t column = (task / cycles) * kTile + threadIdx.x;
    if (column >= row_vectors) continue;
    int64_t positions[Period];
    Vector values[Period][kItemsPerThread] = {};
    positions[0] = leaders[task % cycles];
    const int64_t mask = (int64_t{1} << cp_shift) - 1;
#pragma unroll
    for (int index = 1; index < Period; ++index) {
      const int64_t previous = positions[index - 1];
      positions[index] =
          ((previous & mask) << local_shift) + (previous >> cp_shift);
    }
#pragma unroll
    for (int index = 0; index < Period; ++index) {
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        const int64_t offset = column + item * kCycleThreads;
        if (offset < row_vectors) {
          values[index][item] = data[positions[index] * row_vectors + offset];
        }
      }
    }
#pragma unroll
    for (int index = 0; index < Period; ++index) {
#pragma unroll
      for (int item = 0; item < kItemsPerThread; ++item) {
        const int64_t offset = column + item * kCycleThreads;
        if (offset < row_vectors) {
          data[positions[index] * row_vectors + offset] =
              values[(index + 1) % Period][item];
        }
      }
    }
  }
}

// General shapes, arbitrary CP, and unaligned rows use the same row cycles.
// Only one row tile is retained while the other rows rotate into it. The byte
// path preserves odd widths and unaligned storage offsets without typed loads.
template <typename Element, int Items>
__global__ __launch_bounds__(kGenericThreads) void general_cycles(
    Element* data, const int64_t* leaders, int64_t cycles, int64_t row_elements,
    int64_t local_rows, int64_t cp_size) {
  constexpr int64_t kTile = kGenericThreads * Items;
  const int64_t tiles = (row_elements + kTile - 1) / kTile;
  for (int64_t task = blockIdx.x; task < cycles * tiles; task += gridDim.x) {
    const int64_t leader = leaders[task / tiles];
    if (leader < 0 || leader >= local_rows * cp_size) continue;
    const int64_t column = (task % tiles) * kTile + threadIdx.x;
    Element first[Items] = {};
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      const int64_t offset = column + item * kGenericThreads;
      if (offset < row_elements) first[item] = data[leader * row_elements + offset];
    }
    int64_t current = leader;
    int64_t next = (current % cp_size) * local_rows + current / cp_size;
    while (next != leader) {
#pragma unroll
      for (int item = 0; item < Items; ++item) {
        const int64_t offset = column + item * kGenericThreads;
        if (offset < row_elements) {
          data[current * row_elements + offset] = data[next * row_elements + offset];
        }
      }
      current = next;
      next = (current % cp_size) * local_rows + current / cp_size;
    }
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      const int64_t offset = column + item * kGenericThreads;
      if (offset < row_elements) data[current * row_elements + offset] = first[item];
    }
  }
}

}  // namespace

void interleave_rows_(at::Tensor data, int64_t cp_size, const at::Tensor& leaders) {
  TORCH_CHECK(data.is_cuda() && data.is_contiguous() && data.dim() >= 1,
              "contiguous GPU row tensor required");
  TORCH_CHECK(!data.requires_grad(), "inference-only tensor required");
  TORCH_CHECK(cp_size > 0 && data.size(0) % cp_size == 0,
              "rows must be divisible by positive CP");
  TORCH_CHECK(leaders.is_cuda() && leaders.device() == data.device() &&
                  leaders.scalar_type() == at::kLong && leaders.dim() == 1 &&
                  leaders.is_contiguous(),
              "contiguous int64 cycle plan on the same device required");
  if (data.numel() == 0 || leaders.numel() == 0) return;
  const c10::cuda::CUDAGuard guard(data.device());
  const auto stream = at::cuda::getCurrentCUDAStream(data.get_device());
  const int64_t rows = data.size(0);
  const int64_t row_bytes = data.nbytes() / rows;
  const uintptr_t pointer = reinterpret_cast<uintptr_t>(data.data_ptr());
  if (row_bytes % sizeof(Vector) == 0 && pointer % alignof(Vector) == 0 &&
      (rows & (rows - 1)) == 0 && (cp_size & (cp_size - 1)) == 0 &&
      cp_size > 1 && rows > cp_size) {
    const int total_shift = __builtin_ctzll(rows);
    const int cp_shift = __builtin_ctzll(cp_size);
    const int period = total_shift / std::gcd(total_shift, cp_shift);
    const int64_t vectors = row_bytes / sizeof(Vector);
    const int64_t tile = kCycleThreads * kItemsPerThread;
    const int64_t blocks =
        std::min(kMaxBlocks, leaders.numel() * ((vectors + tile - 1) / tile));
#define LAUNCH_PERIOD(N)                                                       \
    case N:                                                                   \
      short_cycles<N><<<blocks, kCycleThreads, 0, stream>>>(                    \
          static_cast<Vector*>(data.data_ptr()), leaders.data_ptr<int64_t>(), \
          leaders.numel(), vectors, cp_shift, total_shift - cp_shift);         \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                          \
      return
    switch (period) {
      LAUNCH_PERIOD(2);
      LAUNCH_PERIOD(3);
      LAUNCH_PERIOD(4);
      LAUNCH_PERIOD(5);
      LAUNCH_PERIOD(6);
      LAUNCH_PERIOD(7);
      LAUNCH_PERIOD(8);
      LAUNCH_PERIOD(9);
      LAUNCH_PERIOD(10);
      LAUNCH_PERIOD(11);
      LAUNCH_PERIOD(12);
      LAUNCH_PERIOD(13);
      LAUNCH_PERIOD(14);
      LAUNCH_PERIOD(15);
      LAUNCH_PERIOD(16);
    }
#undef LAUNCH_PERIOD
  }
  if (row_bytes % sizeof(uint32_t) == 0 && pointer % alignof(uint32_t) == 0) {
    const int64_t words = row_bytes / sizeof(uint32_t);
    const int64_t tile = kGenericThreads * kGenericItems;
    const int64_t blocks =
        std::min(kMaxBlocks, leaders.numel() * ((words + tile - 1) / tile));
    general_cycles<uint32_t, kGenericItems>
        <<<blocks, kGenericThreads, 0, stream>>>(
            static_cast<uint32_t*>(data.data_ptr()), leaders.data_ptr<int64_t>(),
            leaders.numel(), words, rows / cp_size, cp_size);
  } else {
    const int64_t blocks = std::min(
        kMaxBlocks, leaders.numel() * ((row_bytes + kGenericThreads - 1) / kGenericThreads));
    general_cycles<uint8_t, 1><<<blocks, kGenericThreads, 0, stream>>>(
        static_cast<uint8_t*>(data.data_ptr()), leaders.data_ptr<int64_t>(),
        leaders.numel(), row_bytes, rows / cp_size, cp_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace lightop::cp_output
