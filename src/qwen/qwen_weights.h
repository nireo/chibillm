#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include "metal/metal_context.h"
#include "metal/metal_tensor.h"
#include "model_format/safetensors.h"
#include "qwen/qwen_configs.h"
#include "result.h"

namespace chibillm {

enum class qwen_weights_errc : std::uint8_t {
    missing_tensor,
    unsupported_dtype,
    tensor_shape_mismatch,
    unexpected_tensor_count,
    tensor_count_overflow,
    invalid_configuration,
    tensor_creation_failed,
    metal_allocation_failed,
    tensor_read_failed,
};

struct qwen_layer_weights {
    metal_tensor input_norm;
    metal_tensor post_attention_norm;
    metal_tensor query_norm;
    metal_tensor key_norm;
    // q | k | v projection matrices stacked so one kernel launch reads them all.
    metal_tensor qkv_packed;
    metal_tensor attention_output;
    // mlp gate | up projections stacked for the same reason.
    metal_tensor gateup_packed;
    metal_tensor mlp_down;
};

struct qwen_weights {
    metal_tensor token_embedding;
    metal_tensor final_norm;
    metal_tensor output;
    std::vector<qwen_layer_weights> layers;
};

struct qwen3_5_full_attention_weights {
    metal_tensor query_norm;
    metal_tensor key_norm;
    // gated query | key | value projections packed into one allocation.
    metal_tensor qkv_packed;
    metal_tensor output;
};

struct qwen3_5_linear_attention_weights {
    metal_tensor qkv_projection;
    metal_tensor gate_projection;
    metal_tensor decay_projection;
    metal_tensor learning_rate_projection;
    metal_tensor convolution;
    metal_tensor decay_log;
    metal_tensor learning_rate_bias;
    metal_tensor norm;
    metal_tensor output;
};

using qwen3_5_mixer_weights =
    std::variant<qwen3_5_linear_attention_weights, qwen3_5_full_attention_weights>;

struct qwen3_5_layer_weights {
    metal_tensor input_norm;
    metal_tensor post_attention_norm;
    // mlp gate | up projections use the same packed representation as Qwen3.
    metal_tensor gateup_packed;
    metal_tensor mlp_down;
    qwen3_5_mixer_weights mixer;
};

struct qwen3_5_weights {
    metal_tensor token_embedding;
    metal_tensor final_norm;
    std::vector<qwen3_5_layer_weights> layers;
};

[[nodiscard]] result<void, qwen_weights_errc> validate_qwen_weights(const safetensors_file& weights,
                                                                    const qwen3_config& config);

[[nodiscard]] result<qwen_weights, qwen_weights_errc> load_qwen_weights(
    const metal_context& context, const safetensors_file& file, const qwen3_config& config);

[[nodiscard]] result<void, qwen_weights_errc>
validate_qwen3_5_weights(const safetensors_file& weights, const qwen3_5_config& config);

[[nodiscard]] result<qwen3_5_weights, qwen_weights_errc> load_qwen3_5_weights(
    const metal_context& context, const safetensors_file& file, const qwen3_5_config& config);

} // namespace chibillm
