#include "model_format/weight_reader.h"
#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include "metal/metal_context.h"
#include "model_format/safetensors.h"
#include "qwen/qwen_configs.h"
#include "result.h"
#include "tensor/matrix.h"

namespace chibillm {

struct qwen_layer_weights {
    metal_tensor input_norm;
    metal_tensor post_attention_norm;
    metal_tensor query_norm;
    metal_tensor key_norm;
    // q | k | v projection matrices stacked so one kernel launch reads them all.
    matrix_weight qkv_packed;
    matrix_weight attention_output;
    // mlp gate | up projections stacked for the same reason.
    matrix_weight gateup_packed;
    matrix_weight mlp_down;
};

struct qwen_weights {
    matrix_weight token_embedding;
    metal_tensor final_norm;
    matrix_weight output;
    std::vector<qwen_layer_weights> layers;
};

struct qwen3_5_full_attention_weights {
    metal_tensor query_norm;
    metal_tensor key_norm;
    // gated query | key | value projections packed into one allocation.
    matrix_weight qkv_packed;
    matrix_weight output;
};

struct qwen3_5_linear_attention_weights {
    matrix_weight qkv_projection;
    matrix_weight gate_projection;
    matrix_weight decay_projection;
    matrix_weight learning_rate_projection;
    metal_tensor convolution;
    metal_tensor decay_log;
    metal_tensor learning_rate_bias;
    metal_tensor norm;
    matrix_weight output;
};

using qwen3_5_mixer_weights =
    std::variant<qwen3_5_linear_attention_weights, qwen3_5_full_attention_weights>;

struct qwen3_5_layer_weights {
    metal_tensor input_norm;
    metal_tensor post_attention_norm;
    // mlp gate | up projections use the same packed representation as Qwen3.
    matrix_weight gateup_packed;
    matrix_weight mlp_down;
    qwen3_5_mixer_weights mixer;
};

struct qwen3_5_weights {
    matrix_weight token_embedding;
    metal_tensor final_norm;
    std::vector<qwen3_5_layer_weights> layers;
};

[[nodiscard]] result<void, weight_error> validate_qwen_weights(const safetensors_file& weights,
                                                               const qwen3_config& config);

[[nodiscard]] result<qwen_weights, weight_error>
load_qwen_weights(const metal_context& context,
                  const safetensors_file& file,
                  const qwen3_config& config,
                  weight_quantization quantization = weight_quantization::none);

[[nodiscard]] result<void, weight_error> validate_qwen3_5_weights(const safetensors_file& weights,
                                                                  const qwen3_5_config& config);

[[nodiscard]] result<qwen3_5_weights, weight_error>
load_qwen3_5_weights(const metal_context& context,
                     const safetensors_file& file,
                     const qwen3_5_config& config,
                     weight_quantization quantization = weight_quantization::none);

} // namespace chibillm
