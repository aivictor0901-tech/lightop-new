// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>
#include <torch/extension.h>

namespace lightop {
namespace deep_gemm {

void spec_indexer_logits(const torch::Tensor& q,
                         const torch::Tensor& summary,
                         const torch::Tensor& block_table,
                         const torch::Tensor& weights,
                         const torch::Tensor& nscore,
                         const torch::Tensor& query_start_loc,
                         torch::Tensor& out,
                         double weight_scale);

torch::Tensor get_paged_mqa_logits_metadata(const torch::Tensor& context_lens, int block_kv, int num_sms);

torch::Tensor paged_mqa_logits(const torch::Tensor& q,
                               const torch::Tensor& fused_kv_cache,
                               const torch::Tensor& weights,
                               const torch::Tensor& context_lens,
                               const torch::Tensor& block_table,
                               std::optional<torch::Tensor>& schedule_meta,
                               const int& max_context_len,
                               const bool& clean_logits,
                               const bool& is_prefill = false);

}
}
