// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include "paged_mqa_logits.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_runtime.h>
#include <array>
#include <limits>

#pragma clang fp contract(off)

namespace lightop::deep_gemm {
namespace {
using i2 = int __attribute__((ext_vector_type(2)));
using f4 = float __attribute__((ext_vector_type(4)));
using u8 = uint8_t __attribute__((ext_vector_type(8)));
using u16 = uint8_t __attribute__((ext_vector_type(16)));
using b16 = uint16_t __attribute__((ext_vector_type(16)));
using i4 = int32_t __attribute__((ext_vector_type(4)));
union Bytes16 {
  u16 all;
  struct {
    u8 lo, hi;
  };
};

// Q is already E4M3-valued in BF16 storage. This exact conversion includes
// E4M3 subnormals and signed zero; it is not a second dynamic quantization.
__device__ __forceinline__ uint8_t prerounded_bf16_to_fp8(uint16_t bits) {
  const uint16_t mag = bits & 0x7fff;
  const uint8_t sign = (bits >> 8) & 0x80;
  if (mag >= 0x7f80) return sign | 0x7f;
  if (mag < 0x3c80) {
    union {
      uint32_t u;
      float f;
    } x{uint32_t(mag) << 16};
    return sign | static_cast<uint8_t>(x.f * 512.f);
  }
  return sign | static_cast<uint8_t>((mag >> 4) - 0x3c0);
}

__device__ __forceinline__ u16 pack_prerounded_q(const uint16_t* ptr) {
  const b16 raw = *reinterpret_cast<const b16*>(ptr);
  i4 packed{};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    union {
      uint32_t u;
      float f;
    } a{uint32_t(raw[4 * j]) << 16}, b{uint32_t(raw[4 * j + 1]) << 16},
        c{uint32_t(raw[4 * j + 2]) << 16}, d{uint32_t(raw[4 * j + 3]) << 16};
#if defined(__gfx938__)
    int value = 0;
    value = __builtin_hcu_cvt_pk_fp8_f32(a.f, b.f, value, false);
    value = __builtin_hcu_cvt_pk_fp8_f32(c.f, d.f, value, true);
    packed[j] = value;
#else
    packed[j] = prerounded_bf16_to_fp8(raw[4 * j]) |
                (uint32_t(prerounded_bf16_to_fp8(raw[4 * j + 1])) << 8) |
                (uint32_t(prerounded_bf16_to_fp8(raw[4 * j + 2])) << 16) |
                (uint32_t(prerounded_bf16_to_fp8(raw[4 * j + 3])) << 24);
#endif
  }
  return *reinterpret_cast<u16*>(&packed);
}

__device__ __forceinline__ f4 dot_fp8(u8 a, u8 b, f4 c) {
#if defined(__gfx938__)
  return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(
      *reinterpret_cast<i2*>(&a), *reinterpret_cast<i2*>(&b), c, true, false);
#else
  return c;
#endif
}

// Fixed four-wave128-key CTA for every batch size. Stage Q once, then reuse
// each Q operand for two16-key MMAC accumulators per wave. LIT keeps the
// four heads in registers; this entry is separate from generic MQA kernels.
__global__ __launch_bounds__(256) void spec_indexer_fp8_kernel(
    const uint16_t* q, const uint8_t* pool, const int32_t* bt,
    const at::BFloat16* weights, const int32_t* nscore, const int32_t* qstarts,
    float* out, int64_t qstride, int64_t wstride, int64_t btstride, int64_t ostride,
    int width, float wscale) {
  constexpr int D = 256, KEYS = 128;
  __shared__ __align__(16) uint8_t qs[16 * D];
  const int req = blockIdx.y, first_key = blockIdx.x * KEYS;
  const int qstart = qstarts[req], qlen = qstarts[req + 1] - qstart;
  int widest = 0;
#pragma unroll
  for (int t = 0; t < 4; ++t)
    if (t < qlen) widest = max(widest, nscore[qstart + t]);
  if (first_key >= widest) return;
  const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64;
  const int qr = tid / 16, d0 = (tid % 16) * 16;
  u16 qb{};
  if (qr / 4 < qlen)
    qb = pack_prerounded_q(q + int64_t(qstart + qr / 4) * qstride + (qr % 4) * D + d0);
  *reinterpret_cast<u16*>(qs + qr * D + (d0 ^ ((qr & 15) * 16))) = qb;
  __syncthreads();
  int64_t physical[2];
  bool valid[2];
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const int key = first_key + wave * 32 + n * 16 + lane % 16;
    valid[n] = key < widest && key < width;
    physical[n] =
        valid[n] ? int64_t(bt[int64_t(req) * btstride + key / 8]) * 8 + key % 8 : 0;
  }
  // Prefetch the complete fixed D256 operand set before its consumers.
  // More independent VMEM requests can overlap instead of four serial waits.
  Bytes16 query_fragments[4], key_fragments[2][4];
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) {
    const int off = (lane / 16) * 16 + kb * 64, qrow = lane % 16;
    query_fragments[kb].all =
        *reinterpret_cast<const u16*>(qs + qrow * D + (off ^ (qrow * 16)));
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      key_fragments[n][kb].all = u16{};
      if (valid[n])
        key_fragments[n][kb].all =
            *reinterpret_cast<const u16*>(pool + physical[n] * D + off);
    }
  }
  __builtin_amdgcn_sched_barrier(0);
  f4 acc[2]{};
#pragma unroll
  for (int kb = 0; kb < 4; ++kb) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      acc[n] = dot_fp8(key_fragments[n][kb].lo, query_fragments[kb].lo, acc[n]);
      acc[n] = dot_fp8(key_fragments[n][kb].hi, query_fragments[kb].hi, acc[n]);
    }
  }
  // LIT places QH columns4*g+[0..3] in four adjacent registers:
  // lane%16 owns a key, lane/16 owns a query. Preserve ordered head sum.
  const int token = lane / 16;
  const bool query_valid = token < qlen;
  f4 head_weights{};
  if (query_valid) {
    using W4 = uint16_t __attribute__((ext_vector_type(4)));
    const W4 packed =
        *reinterpret_cast<const W4*>(weights + int64_t(qstart + token) * wstride);
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      union {
        uint32_t u;
        float f;
      } widened{uint32_t(packed[h]) << 16};
      head_weights[h] = widened.f * wscale;
    }
  }
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const int key = first_key + wave * 32 + n * 16 + lane % 16;
    float sum = 0.f;
#pragma unroll
    for (int h = 0; h < 4; ++h) sum += fmaxf(acc[n][h], 0.f) * head_weights[h];
    if (query_valid && key < width && key < nscore[qstart + token])
      out[int64_t(qstart + token) * ostride + key] = sum;
  }
}
}  // namespace

void spec_indexer_logits(const torch::Tensor& q, const torch::Tensor& summary,
                         const torch::Tensor& block_table, const torch::Tensor& weights,
                         const torch::Tensor& nscore,
                         const torch::Tensor& query_start_loc, torch::Tensor& out,
                         double weight_scale) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 4 &&
                  q.size(1) == 1 && q.size(2) == 4 && q.size(3) == 256 &&
                  q.stride(3) == 1 && q.stride(2) == 256 && q.stride(0) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(q.data_ptr()) % 32 == 0,
              "expected aligned BF16 Q[T,1,4,256]");
  TORCH_CHECK(summary.scalar_type() == at::kFloat8_e4m3fn && summary.is_contiguous() &&
                  summary.dim() == 3 && summary.size(1) == 1 &&
                  summary.size(2) == 256 && summary.size(0) % 8 == 0,
              "expected page8 FP8 summary[rows,1,256]");
  TORCH_CHECK(weights.scalar_type() == at::kBFloat16 && weights.dim() == 3 &&
                  weights.size(0) == q.size(0) && weights.size(1) == 1 &&
                  weights.size(2) == 4 && weights.stride(2) == 1,
              "expected row-strided BF16 weights[T,1,4]");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && block_table.dim() == 2 &&
                  block_table.stride(1) == 1 &&
                  query_start_loc.scalar_type() == at::kInt &&
                  query_start_loc.dim() == 1 && query_start_loc.is_contiguous() &&
                  query_start_loc.numel() == block_table.size(0) + 1,
              "invalid request metadata");
  TORCH_CHECK(nscore.scalar_type() == at::kInt && nscore.dim() == 1 &&
                  nscore.is_contiguous() && nscore.numel() == q.size(0),
              "invalid nscore");
  TORCH_CHECK(out.scalar_type() == at::kFloat && out.dim() == 2 &&
                  out.size(0) == q.size(0) && out.stride(1) == 1 &&
                  out.size(1) <= block_table.size(1) * 8,
              "invalid FP32 output");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(summary.data_ptr()) % 16 == 0,
              "expected16-byte-aligned FP8 summary");
  TORCH_CHECK(weights.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(weights.data_ptr()) % 8 == 0,
              "expected aligned row-strided BF16 weights with stride%4=0");
  TORCH_CHECK(out.size(0) < 2 || out.stride(0) >= out.size(1),
              "expected non-overlapping output rows");
  TORCH_CHECK(q.size(0) <= std::numeric_limits<int32_t>::max() &&
                  out.size(1) <= std::numeric_limits<int32_t>::max() &&
                  block_table.size(0) <= 65535,
              "indexer dimensions exceed supported launch/index limits");
  at::assert_no_overlap(out, q);
  at::assert_no_overlap(out, summary);
  at::assert_no_overlap(out, block_table);
  at::assert_no_overlap(out, weights);
  at::assert_no_overlap(out, nscore);
  at::assert_no_overlap(out, query_start_loc);
  for (const auto* x : std::array<const torch::Tensor*, 6>{
           &summary, &block_table, &weights, &nscore, &query_start_loc, &out})
    TORCH_CHECK(x->device() == q.device(), "all tensors must share a CUDA device");
  const c10::cuda::CUDAGuard guard(q.device());
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(std::string(props->gcnArchName).rfind("gfx938", 0) == 0,
              "FP8 indexer requires gfx938");
  if (!q.size(0) || !out.size(1) || !block_table.size(0)) return;
  dim3 grid((out.size(1) + 127) / 128, block_table.size(0));
  hipLaunchKernelGGL(
      spec_indexer_fp8_kernel, grid, dim3(256), 0, at::cuda::getCurrentCUDAStream(),
      reinterpret_cast<const uint16_t*>(q.data_ptr()),
      reinterpret_cast<const uint8_t*>(summary.data_ptr()),
      block_table.data_ptr<int32_t>(), weights.data_ptr<at::BFloat16>(),
      nscore.data_ptr<int32_t>(), query_start_loc.data_ptr<int32_t>(),
      out.data_ptr<float>(), q.stride(0), weights.stride(0), block_table.stride(0),
      out.stride(0), static_cast<int>(out.size(1)), static_cast<float>(weight_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace lightop::deep_gemm
