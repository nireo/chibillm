#include "tensor/attention.h"
#include "tensor/deltanet.h"
#include "tensor/tensor_ops.h"

namespace chibillm {
namespace {

template <typename Operation>
result<metal_tensor, tensor_op_error>
make_output(const metal_context& context, std::vector<std::size_t> dimensions, Operation operation)
{
    auto output = allocate_tensor(context, dtype::f32, std::move(dimensions));
    if (!output)
        return fail(output.error());
    CL_TRY(operation(*output));
    return std::move(*output);
}

template <typename Operation>
result<metal_tensor, tensor_op_error>
make_output_like(const metal_context& context, const metal_tensor& input, Operation operation)
{
    const auto& shape = input.descriptor().shape();
    if (shape.rank() != 2)
        return fail(tensor_op_errc::invalid_rank);
    const auto dimensions = shape.dimensions();
    return make_output(context, { dimensions[0], dimensions[1] }, operation);
}

result<std::array<std::size_t, 2>, tensor_op_error>
projection_shape(const metal_tensor& input, const metal_tensor& weight)
{
    const auto& input_shape = input.descriptor().shape();
    const auto& weight_shape = weight.descriptor().shape();
    if (input_shape.rank() != 2 || weight_shape.rank() != 2)
        return fail(tensor_op_errc::invalid_rank);
    if (input.descriptor().type() != dtype::f32 || weight.descriptor().type() != dtype::bf16)
        return fail(tensor_op_errc::unsupported_dtype);
    if (input_shape.dimensions()[1] != weight_shape.dimensions()[1])
        return fail(tensor_op_errc::inner_dimension_mismatch);
    return std::array { input_shape.dimensions()[0], weight_shape.dimensions()[0] };
}

// Subtracting each width also rejects overflow and zero-width outputs before allocation.
result<void, tensor_op_error>
validate_widths(std::size_t total, std::initializer_list<std::size_t> widths)
{
    for (const auto width : widths) {
        if (width == 0 || width > total)
            return fail(tensor_op_errc::inner_dimension_mismatch);
        total -= width;
    }
    if (total != 0)
        return fail(tensor_op_errc::inner_dimension_mismatch);
    return {};
}

} // namespace

result<metal_tensor, tensor_op_error>
linear(const metal_context& context, const metal_tensor& input, const metal_tensor& weight)
{
    auto shape = projection_shape(input, weight);
    if (!shape)
        return fail(shape.error());
    return make_output(context, { (*shape)[0], (*shape)[1] }, [&](metal_tensor& output) {
        return linear_split(context, input, weight, { &output });
    });
}

result<metal_tensor, tensor_op_error>
linear_add(const metal_context& context,
           const metal_tensor& input,
           const metal_tensor& weight,
           const metal_tensor& residual)
{
    auto shape = projection_shape(input, weight);
    if (!shape)
        return fail(shape.error());
    return make_output(context, { (*shape)[0], (*shape)[1] }, [&](metal_tensor& output) {
        return linear_add(context, input, weight, residual, output);
    });
}

result<std::array<metal_tensor, 2>, tensor_op_error>
linear_split(const metal_context& context,
             const metal_tensor& input,
             const metal_tensor& packed_weight,
             std::size_t first_width,
             std::size_t second_width)
{
    auto shape = projection_shape(input, packed_weight);
    if (!shape)
        return fail(shape.error());
    CL_TRY(validate_widths((*shape)[1], { first_width, second_width }));
    auto first = allocate_tensor(context, dtype::f32, { (*shape)[0], first_width });
    if (!first)
        return fail(first.error());
    auto second = allocate_tensor(context, dtype::f32, { (*shape)[0], second_width });
    if (!second)
        return fail(second.error());
    CL_TRY(linear_split(context, input, packed_weight, { &*first, &*second }));
    return std::array<metal_tensor, 2> { std::move(*first), std::move(*second) };
}

result<std::array<metal_tensor, 3>, tensor_op_error>
linear_split(const metal_context& context,
             const metal_tensor& input,
             const metal_tensor& packed_weight,
             std::size_t first_width,
             std::size_t second_width,
             std::size_t third_width)
{
    auto shape = projection_shape(input, packed_weight);
    if (!shape)
        return fail(shape.error());
    CL_TRY(validate_widths((*shape)[1], { first_width, second_width, third_width }));
    auto first = allocate_tensor(context, dtype::f32, { (*shape)[0], first_width });
    if (!first)
        return fail(first.error());
    auto second = allocate_tensor(context, dtype::f32, { (*shape)[0], second_width });
    if (!second)
        return fail(second.error());
    auto third = allocate_tensor(context, dtype::f32, { (*shape)[0], third_width });
    if (!third)
        return fail(third.error());
    CL_TRY(linear_split(context, input, packed_weight, { &*first, &*second, &*third }));
    return std::array<metal_tensor, 3> { std::move(*first), std::move(*second), std::move(*third) };
}

result<metal_tensor, tensor_op_error>
rms_norm(const metal_context& context,
         const metal_tensor& input,
         const metal_tensor& weight,
         float epsilon,
         bool zero_centered)
{
    return make_output_like(context, input, [&](metal_tensor& output) {
        return rms_norm(context, input, weight, epsilon, output, zero_centered);
    });
}

result<metal_tensor, tensor_op_error>
silu_mul(const metal_context& context, const metal_tensor& gate, const metal_tensor& up)
{
    return make_output_like(
        context, gate, [&](metal_tensor& output) { return silu_mul(context, gate, up, output); });
}

result<std::array<metal_tensor, 2>, tensor_op_error>
split_heads(const metal_context& context, const metal_tensor& input, std::size_t head_count)
{
    const auto& shape = input.descriptor().shape();
    if (shape.rank() != 2)
        return fail(tensor_op_errc::invalid_rank);
    if (input.descriptor().type() != dtype::f32)
        return fail(tensor_op_errc::unsupported_dtype);
    if (head_count == 0)
        return fail(tensor_op_errc::invalid_head_count);
    const auto width = shape.dimensions()[1];
    if (width % head_count != 0 || (width / head_count) % 2 != 0)
        return fail(tensor_op_errc::invalid_head_dimension);
    auto first = allocate_tensor(context, dtype::f32, { shape.dimensions()[0], width / 2 });
    if (!first)
        return fail(first.error());
    auto second = allocate_tensor(context, dtype::f32, { shape.dimensions()[0], width / 2 });
    if (!second)
        return fail(second.error());
    CL_TRY(split_heads(context, input, head_count, *first, *second));
    return std::array<metal_tensor, 2> { std::move(*first), std::move(*second) };
}

result<metal_tensor, tensor_op_error>
rope(const metal_context& context,
     const metal_tensor& input,
     const metal_tensor& positions,
     std::size_t head_count,
     float theta,
     std::size_t rotary_dimension)
{
    return make_output_like(context, input, [&](metal_tensor& output) {
        return rope(context, input, positions, head_count, theta, output, rotary_dimension);
    });
}

result<metal_tensor, tensor_op_error>
paged_attention(const metal_context& context,
                const metal_tensor& queries,
                const prepared_attention_batch& metadata,
                std::size_t layer,
                std::size_t query_head_count,
                const metal_kv_cache& cache)
{
    return make_output_like(context, queries, [&](metal_tensor& output) {
        return paged_attention(context, queries, metadata, layer, query_head_count, cache, output);
    });
}

result<metal_tensor, tensor_op_error>
rms_norm_gated(const metal_context& context,
               const metal_tensor& input,
               const metal_tensor& gate,
               const metal_tensor& weight,
               float epsilon)
{
    return make_output_like(context, input, [&](metal_tensor& output) {
        return rms_norm_gated(context, input, gate, weight, epsilon, output);
    });
}

} // namespace chibillm
