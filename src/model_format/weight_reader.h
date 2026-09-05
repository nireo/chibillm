#pragma once
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
    friend result<weight_bundle, weight_errc> read_weights(const metal_context&,
                                                           const safetensors_file&,
                                                           std::string_view,
                                                           const weight_layout&);
    std::unordered_map<std::string, metal_tensor> tensors_;
};

result<void, weight_errc> validate_weights(const safetensors_file& file,
                                           std::string_view prefix,
                                           const weight_layout& layout);
result<weight_bundle, weight_errc> read_weights(const metal_context& context,
                                                const safetensors_file& file,
                                                std::string_view prefix,
                                                const weight_layout& layout);
} // namespace chibillm
