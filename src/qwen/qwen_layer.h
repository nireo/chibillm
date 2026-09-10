#include "tensor/attention.h"
#include "tensor/tensor_ops.h"
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "metal/metal_context.h"
#include "metal/metal_kv_cache.h"
#include "metal/metal_tensor.h"
#include "qwen/qwen_configs.h"
#include "qwen/qwen_weights.h"
#include "result.h"

namespace chibillm {

struct qwen_qkv {
    metal_tensor query;
    metal_tensor key;
    metal_tensor value;
};

[[nodiscard]] result<qwen_qkv, tensor_op_errc> project_qwen_qkv(const metal_context& context,
                                                                const qwen3_config& config,
                                                                const qwen_layer_weights& weights,
                                                                const metal_tensor& hidden_states);

[[nodiscard]] result<qwen_qkv, tensor_op_errc> normalize_qwen_qk(const metal_context& context,
                                                                 const qwen3_config& config,
                                                                 const qwen_layer_weights& weights,
                                                                 qwen_qkv qkv);

[[nodiscard]] result<metal_tensor, tensor_op_errc> run_qwen_layers(const metal_context& context,
                                                                   const qwen3_config& config,
                                                                   const qwen_weights& weights,
                                                                   metal_tensor hidden_states,
                                                                   attention_metadata metadata,
                                                                   metal_kv_cache& cache);

class qwen3_5_model_state;
struct model_batch;

// Hybrid text decoder layers. Rows follow batch.items' contiguous token ranges;
// each sequence's convolution and recurrent state is selected by its stable ID.
// The caller must successfully begin_batch(batch) before execution and owns the
// compute pass. Finish GPU work before commit_batch(), or abort_batch() on any
// failure (including failures after these layers). This function never commits.
[[nodiscard]] result<metal_tensor, tensor_op_errc>
run_qwen3_5_layers(const metal_context& context,
                   const qwen3_5_config& config,
                   const qwen3_5_weights& weights,
                   metal_tensor hidden_states,
                   const model_batch& batch,
                   qwen3_5_model_state& state);

// Text-only full-attention mixer, including input norm and the residual addition.
// cache_layer is the compact KV layer index from qwen3_5_model_state::cache_layer.
// Metadata must remain alive through execution. The caller owns the compute pass.
[[nodiscard]] result<metal_tensor, tensor_op_errc>
run_qwen3_5_full_attention(const metal_context& context,
                           const qwen3_5_config& config,
                           const qwen3_5_layer_weights& weights,
                           const metal_tensor& hidden_states,
                           const metal_tensor& slots,
                           const prepared_attention_batch& prepared,
                           std::size_t cache_layer,
                           metal_kv_cache& cache);

} // namespace chibillm
