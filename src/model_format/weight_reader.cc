#include "model_format/weight_reader.h"
#include <cassert>
#include <limits>

namespace chibillm {
metal_tensor
weight_bundle::take(const std::string& name)
{
    auto node = tensors_.extract(name);
    assert(!node.empty());
    return std::move(node.mapped());
}

result<void, weight_errc>
validate_weights(const safetensors_file& file, std::string_view prefix, const weight_layout& layout)
{
    for (const auto& group : layout) {
        if (group.tensors.empty())
            return fail(weight_errc::invalid_configuration);
        for (const auto& spec : group.tensors) {
            const auto* tensor = file.find(std::string(prefix) + spec.name);
            if (!tensor)
                return fail(weight_errc::missing_tensor);
            if (tensor->type != spec.type)
                return fail(weight_errc::unsupported_dtype);
            if (tensor->shape != spec.shape)
                return fail(weight_errc::tensor_shape_mismatch);
        }
    }
    return {};
}

result<weight_bundle, weight_errc>
read_weights(const metal_context& context,
             const safetensors_file& file,
             std::string_view prefix,
             const weight_layout& layout)
{
    CL_TRY(validate_weights(file, prefix, layout));
    weight_bundle bundle;
    for (const auto& group : layout) {
        const auto& first = group.tensors.front();
        auto shape = first.shape;
        dtype type;
        if (first.type == safetensors_dtype::bf16)
            type = dtype::bf16;
        else if (first.type == safetensors_dtype::f32)
            type = dtype::f32;
        else
            return fail(weight_errc::unsupported_dtype);
        for (std::size_t i = 1; i < group.tensors.size(); ++i) {
            const auto& spec = group.tensors[i];
            if (shape.size() != 2
                || spec.shape.size() != 2
                || spec.shape[1] != shape[1]
                || spec.type != safetensors_dtype::bf16
                || type != dtype::bf16) {
                return fail(weight_errc::tensor_shape_mismatch);
            }
            if (shape[0] > std::numeric_limits<std::size_t>::max() - spec.shape[0]) {
                return fail(weight_errc::tensor_count_overflow);
            }
            shape[0] += spec.shape[0];
        }
        auto tensor = metal_tensor::make(context, type, std::move(shape));
        if (!tensor)
            return fail(tensor.error() == metal_tensor_errc::invalid_descriptor
                            ? weight_errc::tensor_creation_failed
                            : weight_errc::metal_allocation_failed);
        std::size_t offset = 0;
        auto bytes = tensor->buffer().bytes();
        for (const auto& spec : group.tensors) {
            const auto name = std::string(prefix) + spec.name;
            const auto size = static_cast<std::size_t>(file.find(name)->byte_count);
            if (!file.read(name, bytes.subspan(offset, size)))
                return fail(weight_errc::tensor_read_failed);
            offset += size;
        }
        bundle.tensors_.emplace(group.name, std::move(*tensor));
    }
    return bundle;
}
} // namespace chibillm
