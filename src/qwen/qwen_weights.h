#pragma once

#include <cstdint>

#include "model_format/safetensors.h"
#include "qwen/qwen_config.h"
#include "result.h"

namespace chibillm {

enum class qwen_weights_errc : std::uint8_t {
    missing_tensor,
    unsupported_dtype,
    tensor_shape_mismatch,
    unexpected_tensor_count,
    tensor_count_overflow,
};

[[nodiscard]] result<void, qwen_weights_errc> validate_qwen_weights(const safetensors_file& weights,
                                                                    const qwen_config& config);

} // namespace chibillm
