#pragma once

#include "error.h"
#include "metal/metal_tensor.h"
#include "model_format/safetensors.h"
#include <unordered_map>

namespace chibillm {
enum class weight_errc : std::uint8_t {
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

[[nodiscard]] inline std::string_view
error_name(weight_errc code) noexcept
{
    static constexpr std::array names {
        "weight.missing_tensor",         "weight.unsupported_dtype",
        "weight.tensor_shape_mismatch",  "weight.unexpected_tensor_count",
        "weight.tensor_count_overflow",  "weight.invalid_configuration",
        "weight.tensor_creation_failed", "weight.metal_allocation_failed",
        "weight.tensor_read_failed",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "weight.unknown_error";
}

using weight_error = error<weight_errc>;

struct tensor_spec {
    std::string name;
    std::vector<std::size_t> shape;
    safetensors_dtype type = safetensors_dtype::bf16;
};

struct weight_group {
    std::string name;
    std::vector<tensor_spec> tensors;
};

using weight_layout = std::vector<weight_group>;

class weight_bundle {
public:
    metal_tensor take(const std::string& name);

private:
    friend result<weight_bundle, weight_error> read_weights(const metal_context&,
                                                            const safetensors_file&,
                                                            std::string_view,
                                                            const weight_layout&);
    std::unordered_map<std::string, metal_tensor> tensors_;
};

result<void, weight_error> validate_weights(const safetensors_file& file,
                                            std::string_view prefix,
                                            const weight_layout& layout);
result<weight_bundle, weight_error> read_weights(const metal_context& context,
                                                 const safetensors_file& file,
                                                 std::string_view prefix,
                                                 const weight_layout& layout);
} // namespace chibillm
