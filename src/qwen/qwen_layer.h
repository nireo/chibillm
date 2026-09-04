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

enum class qwen_layer_errc : std::uint8_t {
    tensor_creation_failed,
    metal_allocation_failed,
    invalid_input,
    backend_failure,
};

struct qwen_qkv {
    metal_tensor query;
    metal_tensor key;
    metal_tensor value;
};

struct qwen_attention_metadata {
    std::span<const std::uint32_t> positions;
    std::span<const std::uint32_t> slots;
    std::span<const std::uint32_t> block_table;
    std::span<const std::uint32_t> block_table_offsets;
    std::span<const std::uint32_t> block_table_lengths;
};

[[nodiscard]] result<qwen_qkv, qwen_layer_errc> project_qwen_qkv(const metal_context& context,
                                                                 const qwen3_config& config,
                                                                 const qwen_layer_weights& weights,
                                                                 const metal_tensor& hidden_states);

[[nodiscard]] result<qwen_qkv, qwen_layer_errc> normalize_qwen_qk(const metal_context& context,
                                                                  const qwen3_config& config,
                                                                  const qwen_layer_weights& weights,
                                                                  qwen_qkv qkv);

[[nodiscard]] result<metal_tensor, qwen_layer_errc> run_qwen_mlp(const metal_context& context,
                                                                 const qwen3_config& config,
                                                                 const qwen_layer_weights& weights,
                                                                 const metal_tensor& hidden_states);

[[nodiscard]] result<metal_tensor, qwen_layer_errc>
run_qwen_layers(const metal_context& context,
                const qwen3_config& config,
                const qwen_weights& weights,
                metal_tensor hidden_states,
                qwen_attention_metadata metadata,
                metal_kv_cache& cache);

} // namespace chibillm
