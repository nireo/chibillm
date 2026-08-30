#pragma once

#include <cstdint>
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

[[nodiscard]] result<void, qwen_weights_errc> validate_qwen_weights(const safetensors_file& weights,
                                                                    const qwen3_config& config);

[[nodiscard]] result<qwen_weights, qwen_weights_errc> load_qwen_weights(
    const metal_context& context, const safetensors_file& file, const qwen3_config& config);

} // namespace chibillm
