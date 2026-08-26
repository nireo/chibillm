#pragma once

#include <cstdint>

#include "metal/metal_context.h"
#include "metal/metal_tensor.h"
#include "qwen/qwen_config.h"
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

[[nodiscard]] result<qwen_qkv, qwen_layer_errc> project_qwen_qkv(const metal_context& context,
                                                                 const qwen_config& config,
                                                                 const qwen_layer_weights& weights,
                                                                 const metal_tensor& hidden_states);

[[nodiscard]] result<qwen_qkv, qwen_layer_errc> normalize_qwen_qk(const metal_context& context,
                                                                  const qwen_config& config,
                                                                  const qwen_layer_weights& weights,
                                                                  qwen_qkv qkv);

} // namespace chibillm
