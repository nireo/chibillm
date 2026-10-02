#include "tensor/tensor_ops.h"
#include "metal/metal_kernels.h"
#include "tensor/types.h"
#include "tensor/validation.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace chibillm {

namespace {

result<std::size_t, tensor_op_error>
validate_elementwise_output(const metal_tensor& lhs,
                            const metal_tensor& rhs,
                            const metal_tensor& output)
{
    const auto& lhs_descriptor = lhs.descriptor();
    const auto& rhs_descriptor = rhs.descriptor();
    const auto& output_descriptor = output.descriptor();
    const auto& lhs_shape = lhs_descriptor.shape();
    const auto& rhs_shape = rhs_descriptor.shape();
    const auto& output_shape = output_descriptor.shape();

    CL_TRY(validate_tensor_layouts(
        { { lhs, 2, dtype::f32 }, { rhs, 2, dtype::f32 }, { output, 2, dtype::f32 } }));

    if (lhs_shape.dimensions()[0] != rhs_shape.dimensions()[0]
        || lhs_shape.dimensions()[1] != rhs_shape.dimensions()[1]) {
        return fail(tensor_op_errc::input_shape_mismatch);
    }

    if (output_shape.dimensions()[0] != lhs_shape.dimensions()[0]
        || output_shape.dimensions()[1] != lhs_shape.dimensions()[1]) {
        return fail(tensor_op_errc::output_shape_mismatch);
    }

    return lhs_shape.element_count();
}

} // namespace

result<metal_tensor, tensor_op_error>
allocate_tensor(const metal_context& context, dtype type, std::vector<std::size_t> dimensions)
{
    auto tensor = metal_tensor::make(context, type, std::move(dimensions));
    if (!tensor)
        return fail(tensor.error() == metal_tensor_errc::invalid_descriptor
                        ? tensor_op_errc::tensor_creation_failed
                        : tensor_op_errc::allocation_failed,
                    tensor.error());
    return std::move(*tensor);
}

result<metal_tensor, tensor_op_error>
upload_u32(const metal_context& context, std::span<const std::uint32_t> values)
{
    auto tensor = allocate_tensor(context, dtype::u32, { values.size() });
    if (!tensor)
        return fail(tensor.error());
    std::memcpy(tensor->buffer().bytes().data(), values.data(), values.size_bytes());
    return std::move(*tensor);
}

result<void, tensor_op_error>
linear_add(const metal_context& context,
           const metal_tensor& input,
           const metal_tensor& weight,
           const metal_tensor& residual,
           metal_tensor& output)
{
    const auto& input_shape = input.descriptor().shape();
    const auto& weight_shape = weight.descriptor().shape();
    const auto& residual_shape = residual.descriptor().shape();
    const auto& output_shape = output.descriptor().shape();

    CL_TRY(validate_tensor_layouts({ { input, 2, dtype::f32 },
                                     { weight, 2, dtype::bf16 },
                                     { residual, 2, dtype::f32 },
                                     { output, 2, dtype::f32 } }));

    const auto rows = input_shape.dimensions()[0];
    const auto input_features = input_shape.dimensions()[1];
    const auto output_features = weight_shape.dimensions()[0];
    if (weight_shape.dimensions()[1] != input_features) {
        return fail(tensor_op_errc::inner_dimension_mismatch);
    }
    if (residual_shape.dimensions()[0] != rows
        || residual_shape.dimensions()[1] != output_features) {
        return fail(tensor_op_errc::input_shape_mismatch);
    }
    if (output_shape.dimensions()[0] != rows || output_shape.dimensions()[1] != output_features) {
        return fail(tensor_op_errc::output_shape_mismatch);
    }

    if (rows != 1) {
        const std::array<metal_buffer*, 3> output_buffers {
            &output.buffer(),
            &output.buffer(),
            &output.buffer(),
        };
        const std::array<std::size_t, 3> widths { output_features, 0, 0 };
        const auto dispatched = metal_kernels(context).dispatch_linear_split_bf16(
            input.buffer(), weight.buffer(), output_buffers, rows, input_features, widths);
        if (!dispatched) {
            return fail(tensor_op_errc::backend_failure, dispatched.error());
        }
        return add(context, residual, output, output);
    }

    const auto dispatched = metal_kernels(context).dispatch_linear_add_bf16(
        input.buffer(), weight.buffer(), residual.buffer(), output.buffer(), input_features,
        output_features);
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure, dispatched.error());
    }
    return {};
}

result<void, tensor_op_error>
linear_split(const metal_context& context,
             const metal_tensor& input,
             const metal_tensor& packed_weight,
             std::initializer_list<metal_tensor*> outputs)
{
    const auto& input_shape = input.descriptor().shape();
    const auto& weight_shape = packed_weight.descriptor().shape();
    CL_TRY(
        validate_tensor_layouts({ { input, 2, dtype::f32 }, { packed_weight, 2, dtype::bf16 } }));
    if (outputs.size() == 0 || outputs.size() > 3) {
        return fail(tensor_op_errc::output_shape_mismatch);
    }

    const auto rows = input_shape.dimensions()[0];
    const auto input_features = input_shape.dimensions()[1];
    std::array<metal_buffer*, 3> output_buffers;
    std::array<std::size_t, 3> widths {};
    std::size_t i = 0;
    std::size_t total_width = 0;
    for (auto* output : outputs) {
        if (output == nullptr) {
            return fail(tensor_op_errc::output_shape_mismatch);
        }
        const auto& shape = output->descriptor().shape();
        if (shape.rank() != 2) {
            return fail(tensor_op_errc::invalid_rank);
        }
        if (shape.dimensions()[0] != rows) {
            return fail(tensor_op_errc::output_shape_mismatch);
        }
        if (output->descriptor().type() != dtype::f32) {
            return fail(tensor_op_errc::unsupported_dtype);
        }
        widths[i] = shape.dimensions()[1];
        total_width += widths[i];
        output_buffers[i++] = &output->buffer();
    }
    while (i < output_buffers.size()) {
        output_buffers[i++] = output_buffers[0];
    }

    if (weight_shape.dimensions()[0] != total_width
        || weight_shape.dimensions()[1] != input_features) {
        return fail(tensor_op_errc::inner_dimension_mismatch);
    }

    const auto dispatched = metal_kernels(context).dispatch_linear_split_bf16(
        input.buffer(), packed_weight.buffer(), output_buffers, rows, input_features, widths);
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure, dispatched.error());
    }

    return {};
}

result<void, tensor_op_error>
embedding_lookup(const metal_context& context,
                 const metal_tensor& token_ids,
                 const metal_tensor& weight,
                 metal_tensor& output)
{
    const auto& token_shape = token_ids.descriptor().shape();
    const auto& weight_shape = weight.descriptor().shape();
    const auto& output_shape = output.descriptor().shape();

    CL_TRY(validate_tensor_layouts(
        { { token_ids, 1, dtype::i32 }, { weight, 2, dtype::bf16 }, { output, 2, dtype::f32 } }));

    const auto token_count = token_shape.dimensions()[0];
    const auto vocabulary_size = weight_shape.dimensions()[0];
    const auto hidden_size = weight_shape.dimensions()[1];

    if (output_shape.dimensions()[0] != token_count
        || output_shape.dimensions()[1] != hidden_size) {
        return fail(tensor_op_errc::output_shape_mismatch);
    }

    const auto token_bytes = token_ids.buffer().bytes();
    for (std::size_t index = 0; index < token_count; ++index) {
        std::int32_t token = 0;
        std::memcpy(&token, token_bytes.data() + index * sizeof(token), sizeof(token));

        if (token < 0 || static_cast<std::size_t>(token) >= vocabulary_size) {
            return fail(tensor_op_errc::token_out_of_range);
        }
    }

    const auto dispatched = metal_kernels(context).dispatch_embedding_bf16(
        token_ids.buffer(), weight.buffer(), output.buffer(), token_count, hidden_size);
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure, dispatched.error());
    }

    return {};
}

result<void, tensor_op_error>
rms_norm(const metal_context& context,
         const metal_tensor& input,
         const metal_tensor& weight,
         float epsilon,
         metal_tensor& output,
         bool zero_centered)
{
    const auto& input_shape = input.descriptor().shape();
    const auto& weight_shape = weight.descriptor().shape();
    const auto& output_shape = output.descriptor().shape();

    CL_TRY(validate_tensor_layouts(
        { { input, 2, dtype::f32 }, { weight, 1, dtype::bf16 }, { output, 2, dtype::f32 } }));

    const auto rows = input_shape.dimensions()[0];
    const auto hidden_size = input_shape.dimensions()[1];

    const auto group_size = weight_shape.dimensions()[0];
    if (hidden_size % group_size != 0) {
        return fail(tensor_op_errc::inner_dimension_mismatch);
    }

    if (output_shape.dimensions()[0] != rows || output_shape.dimensions()[1] != hidden_size) {
        return fail(tensor_op_errc::output_shape_mismatch);
    }

    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        return fail(tensor_op_errc::invalid_epsilon);
    }

    const auto groups_per_row = hidden_size / group_size;
    const auto dispatched = metal_kernels(context).dispatch_rms_norm_bf16(
        input.buffer(), weight.buffer(), output.buffer(), rows * groups_per_row, group_size,
        epsilon, zero_centered);
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure, dispatched.error());
    }

    return {};
}

namespace {
result<void, tensor_op_error>
gate_mul(const metal_context& context,
         const metal_tensor& gate,
         const metal_tensor& up,
         metal_tensor& output,
         bool sigmoid_only)
{
    auto count = validate_elementwise_output(gate, up, output);
    if (!count)
        return fail(count.error());

    const auto dispatched = metal_kernels(context).dispatch_silu_mul_f32(
        gate.buffer(), up.buffer(), output.buffer(), *count, sigmoid_only);
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure, dispatched.error());
    }

    return {};
}

} // namespace

result<void, tensor_op_error>
silu_mul(const metal_context& context,
         const metal_tensor& gate,
         const metal_tensor& up,
         metal_tensor& output)
{
    return gate_mul(context, gate, up, output, false);
}

result<void, tensor_op_error>
sigmoid_mul(const metal_context& context,
            const metal_tensor& gate,
            const metal_tensor& input,
            metal_tensor& output)
{
    return gate_mul(context, gate, input, output, true);
}

result<void, tensor_op_error>
add(const metal_context& context,
    const metal_tensor& lhs,
    const metal_tensor& rhs,
    metal_tensor& output)
{
    auto count = validate_elementwise_output(lhs, rhs, output);
    if (!count)
        return fail(count.error());

    const auto dispatched = metal_kernels(context).dispatch_add_f32(lhs.buffer(), rhs.buffer(),
                                                                    output.buffer(), *count);
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure, dispatched.error());
    }

    return {};
}

result<void, tensor_op_error>
split_heads(const metal_context& context,
            const metal_tensor& input,
            std::size_t head_count,
            metal_tensor& first,
            metal_tensor& second)
{
    const auto& shape = input.descriptor().shape();
    CL_TRY(validate_tensor_layouts(
        { { input, 2, dtype::f32 }, { first, 2, dtype::f32 }, { second, 2, dtype::f32 } }));

    if (head_count == 0)
        return fail(tensor_op_errc::invalid_head_count);

    const auto dims = shape.dimensions();
    if (dims[1] % head_count != 0 || (dims[1] / head_count) % 2 != 0)
        return fail(tensor_op_errc::invalid_head_dimension);

    for (const auto* output : { &first, &second }) {
        const auto out = output->descriptor().shape().dimensions();
        if (out[0] != dims[0] || out[1] != dims[1] / 2)
            return fail(tensor_op_errc::output_shape_mismatch);
    }

    if (first.buffer().bytes().data() == second.buffer().bytes().data()
        || input.buffer().bytes().data() == first.buffer().bytes().data()
        || input.buffer().bytes().data() == second.buffer().bytes().data())
        return fail(tensor_op_errc::unsupported_aliasing);

    auto dispatched = metal_kernels(context).dispatch_split_heads_f32(
        input.buffer(), first.buffer(), second.buffer(), dims[0], head_count,
        dims[1] / head_count / 2);
    if (!dispatched)
        return fail(tensor_op_errc::backend_failure, dispatched.error());

    return {};
}

result<void, tensor_op_error>
rope(const metal_context& context,
     const metal_tensor& input,
     const metal_tensor& positions,
     std::size_t head_count,
     float theta,
     metal_tensor& output,
     std::size_t rotary_dimension)
{
    const auto& input_shape = input.descriptor().shape();
    const auto& position_shape = positions.descriptor().shape();
    const auto& output_shape = output.descriptor().shape();

    CL_TRY(validate_tensor_layouts(
        { { input, 2, dtype::f32 }, { positions, 1, dtype::u32 }, { output, 2, dtype::f32 } }));

    const auto rows = input_shape.dimensions()[0];
    const auto feature_count = input_shape.dimensions()[1];

    if (position_shape.dimensions()[0] != rows) {
        return fail(tensor_op_errc::position_count_mismatch);
    }

    if (head_count == 0) {
        return fail(tensor_op_errc::invalid_head_count);
    }

    if (feature_count % head_count != 0) {
        return fail(tensor_op_errc::invalid_head_dimension);
    }

    const auto head_dimension = feature_count / head_count;
    if (rotary_dimension == 0)
        rotary_dimension = head_dimension;

    if (head_dimension % 2 != 0 || rotary_dimension % 2 != 0 || rotary_dimension > head_dimension) {
        return fail(tensor_op_errc::invalid_head_dimension);
    }

    if (output_shape.dimensions()[0] != rows || output_shape.dimensions()[1] != feature_count) {
        return fail(tensor_op_errc::output_shape_mismatch);
    }

    if (!std::isfinite(theta) || theta <= 0.0F) {
        return fail(tensor_op_errc::invalid_rope_theta);
    }

    const auto dispatched = metal_kernels(context).dispatch_rope_f32(
        input.buffer(), positions.buffer(), output.buffer(), rows, head_count, head_dimension,
        theta, rotary_dimension);
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure, dispatched.error());
    }

    return {};
}

result<void, tensor_op_error>
store_kv(const metal_context& context,
         const metal_tensor& keys,
         const metal_tensor& values,
         const metal_tensor& slot_mapping,
         std::size_t layer,
         metal_kv_cache& cache)
{
    const auto& key_shape = keys.descriptor().shape();
    const auto& value_shape = values.descriptor().shape();
    const auto& slot_shape = slot_mapping.descriptor().shape();

    CL_TRY(validate_tensor_layouts(
        { { keys, 2, dtype::f32 }, { values, 2, dtype::f32 }, { slot_mapping, 1, dtype::u32 } }));

    const auto rows = key_shape.dimensions()[0];
    const auto feature_count = key_shape.dimensions()[1];
    if (value_shape.dimensions()[0] != rows || value_shape.dimensions()[1] != feature_count) {
        return fail(tensor_op_errc::input_shape_mismatch);
    }

    if (slot_shape.dimensions()[0] != rows) {
        return fail(tensor_op_errc::cache_slot_count_mismatch);
    }

    if (feature_count != cache.elements_per_token()) {
        return fail(tensor_op_errc::cache_feature_count_mismatch);
    }

    if (layer >= cache.layer_count()) {
        return fail(tensor_op_errc::cache_layer_out_of_range);
    }

    const auto slot_count = cache.block_count() * cache.block_size();
    const auto slot_bytes = slot_mapping.buffer().bytes();
    for (std::size_t row = 0; row < rows; ++row) {
        std::uint32_t slot = 0;
        std::memcpy(&slot, slot_bytes.data() + row * sizeof(slot), sizeof(slot));
        if (slot >= slot_count) {
            return fail(tensor_op_errc::cache_slot_out_of_range);
        }
    }

    const auto dispatched = metal_kernels(context).dispatch_store_kv_f32(
        keys.buffer(), values.buffer(), slot_mapping.buffer(), cache.keys().buffer(),
        cache.values().buffer(), rows, feature_count, layer, slot_count);
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure, dispatched.error());
    }

    return {};
}

} // namespace chibillm
