#include "model_format/weight_reader.h"
#include <cassert>
#include <limits>
#include <tuple>

namespace chibillm {
namespace {

std::string
metadata_name(std::string_view weight, std::string_view suffix)
{
    return std::string(weight.substr(0, weight.size() - std::string_view(".weight").size()))
        + std::string(suffix);
}

result<void, weight_error>
validate_tensor(const safetensors_file& file,
                const std::string& name,
                safetensors_dtype type,
                const std::vector<std::size_t>& shape)
{
    const auto* tensor = file.find(name);
    if (!tensor)
        return fail(weight_errc::missing_tensor, name);
    if (tensor->type != type)
        return fail(weight_errc::unsupported_dtype, name);
    if (tensor->shape != shape)
        return fail(weight_errc::tensor_shape_mismatch, name);
    return {};
}

result<quantized_matrix, weight_error>
read_q4(const metal_context& context,
        const safetensors_file& file,
        std::string_view prefix,
        const weight_group& group,
        std::vector<std::size_t> dimensions)
{
    const auto rows = dimensions[0], columns = dimensions[1];
    auto shape = tensor_shape::make(std::move(dimensions));
    if (!shape)
        return fail(weight_errc::tensor_creation_failed, shape.error(), group.name);
    auto packed = metal_tensor::make(context, dtype::u32, { rows, columns / 8 });
    auto scales = metal_tensor::make(context, dtype::bf16, { rows, columns / 64 });
    auto offsets = metal_tensor::make(context, dtype::bf16, { rows, columns / 64 });
    if (!packed)
        return fail(weight_errc::metal_allocation_failed, packed.error(), group.name);
    if (!scales)
        return fail(weight_errc::metal_allocation_failed, scales.error(), group.name);
    if (!offsets)
        return fail(weight_errc::metal_allocation_failed, offsets.error(), group.name);
    std::size_t row = 0;
    for (const auto& spec : group.tensors) {
        const auto name = std::string(prefix) + spec.name;
        for (const auto& [source, destination, byte_offset] :
             { std::tuple { name, &*packed, row * columns / 8 * sizeof(std::uint32_t) },
               std::tuple { metadata_name(name, ".scales"), &*scales,
                            row * columns / 64 * sizeof(std::uint16_t) },
               std::tuple { metadata_name(name, ".biases"), &*offsets,
                            row * columns / 64 * sizeof(std::uint16_t) } }) {
            const auto size = static_cast<std::size_t>(file.find(source)->byte_count);
            auto read = file.read(source, destination->buffer().bytes().subspan(byte_offset, size));
            if (!read)
                return fail(weight_errc::tensor_read_failed, read.error(), source);
        }
        row += spec.shape[0];
    }
    auto matrix = quantized_matrix::from_packed(std::move(*shape), std::move(*packed),
                                                std::move(*scales), std::move(*offsets));
    if (!matrix)
        return fail(weight_errc::quantization_failed, matrix.error(), group.name);
    return std::move(*matrix);
}

} // namespace

metal_tensor
weight_bundle::take_tensor(const std::string& name)
{
    auto node = tensors_.extract(name);
    assert(!node.empty());
    return std::move(node.mapped());
}

matrix_weight
weight_bundle::take_matrix(const std::string& name)
{
    auto node = matrices_.extract(name);
    assert(!node.empty());
    return std::move(node.mapped());
}

result<void, weight_error>
validate_weights(const safetensors_file& file, std::string_view prefix, const weight_layout& layout)
{
    for (const auto& group : layout) {
        if (group.tensors.empty())
            return fail(weight_errc::invalid_configuration);
        const auto* first = file.find(std::string(prefix) + group.tensors.front().name);
        const bool packed = first && first->type == safetensors_dtype::u32;
        for (const auto& spec : group.tensors) {
            if (group.role != weight_role::tensor
                && (spec.shape.size() != 2 || spec.type != safetensors_dtype::bf16))
                return fail(weight_errc::invalid_configuration, group.name);
            const auto name = std::string(prefix) + spec.name;
            const auto* tensor = file.find(name);
            if (!tensor)
                return fail(weight_errc::missing_tensor, name);
            if (packed && group.role != weight_role::tensor) {
                if (!name.ends_with(".weight") || spec.shape[1] % 64 != 0)
                    return fail(weight_errc::invalid_configuration, name);
                CL_TRY(validate_tensor(file, name, safetensors_dtype::u32,
                                       { spec.shape[0], spec.shape[1] / 8 }));
                for (const auto suffix : { ".scales", ".biases" })
                    CL_TRY(validate_tensor(file, metadata_name(name, suffix),
                                           safetensors_dtype::bf16,
                                           { spec.shape[0], spec.shape[1] / 64 }));
            } else {
                CL_TRY(validate_tensor(file, name, spec.type, spec.shape));
                if (group.role != weight_role::tensor
                    && name.ends_with(".weight")
                    && (file.find(metadata_name(name, ".scales"))
                        || file.find(metadata_name(name, ".biases"))))
                    return fail(weight_errc::invalid_configuration, name);
            }
        }
    }
    return {};
}

result<weight_bundle, weight_error>
read_weights(const metal_context& context,
             const safetensors_file& file,
             std::string_view prefix,
             const weight_layout& layout,
             weight_quantization quantization)
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
        if (group.role != weight_role::tensor
            && file.find(std::string(prefix) + first.name)->type == safetensors_dtype::u32) {
            auto matrix = read_q4(context, file, prefix, group, std::move(shape));
            if (!matrix)
                return fail(matrix.error());
            bundle.matrices_.emplace(group.name, std::move(*matrix));
            continue;
        }
        auto tensor = metal_tensor::make(context, type, std::move(shape));
        if (!tensor)
            return fail(tensor.error() == metal_tensor_errc::invalid_descriptor
                            ? weight_errc::tensor_creation_failed
                            : weight_errc::metal_allocation_failed,
                        tensor.error(), group.name);
        std::size_t offset = 0;
        auto bytes = tensor->buffer().bytes();
        for (const auto& spec : group.tensors) {
            const auto name = std::string(prefix) + spec.name;
            const auto size = static_cast<std::size_t>(file.find(name)->byte_count);
            auto read = file.read(name, bytes.subspan(offset, size));
            if (!read)
                return fail(weight_errc::tensor_read_failed, read.error(), name);
            offset += size;
        }
        if (group.role != weight_role::tensor) {
            // Small or unaligned matrices stay dense. Sensitive recurrent control
            // projections explicitly opt out in the model's weight layout.
            if (quantization == weight_quantization::q4
                && group.role == weight_role::matrix
                && tensor->descriptor().shape().dimensions()[0] >= 64
                && tensor->descriptor().shape().dimensions()[1] % quantized_matrix::group_size
                    == 0) {
                auto packed = quantized_matrix::quantize(context, *tensor);
                if (!packed)
                    return fail(weight_errc::quantization_failed, packed.error(), group.name);
                bundle.matrices_.emplace(group.name, std::move(*packed));
            } else {
                bundle.matrices_.emplace(group.name, std::move(*tensor));
            }
        } else {
            bundle.tensors_.emplace(group.name, std::move(*tensor));
        }
    }
    return bundle;
}
} // namespace chibillm
