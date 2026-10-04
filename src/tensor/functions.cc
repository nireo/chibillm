#include "tensor/functions.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "metal/metal_kernels.h"
#include "tensor/attention.h"
#include "tensor/deltanet.h"

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

result<metal_tensor, tensor_op_error>
normalized_swiglu(const metal_context& context,
                  const metal_tensor& norm,
                  const metal_tensor& gateup,
                  const metal_tensor& down,
                  float epsilon,
                  const metal_tensor& hidden_states,
                  bool zero_centered)
{
    if (norm.descriptor().shape().rank() != 1 || down.descriptor().shape().rank() != 2) {
        return fail(tensor_op_errc::input_shape_mismatch);
    }
    const auto hidden_size = norm.descriptor().shape().dimensions()[0];
    const auto intermediate_size = down.descriptor().shape().dimensions()[1];
    const auto& shape = hidden_states.descriptor().shape();
    if (shape.rank() != 2
        || hidden_states.descriptor().type() != dtype::f32
        || shape.dimensions()[1] != hidden_size) {
        return fail(tensor_op_errc::input_shape_mismatch);
    }

    auto normalized = rms_norm(context, hidden_states, norm, epsilon, zero_centered);
    if (!normalized)
        return fail(normalized.error());
    auto projected =
        linear_split(context, *normalized, gateup, intermediate_size, intermediate_size);
    if (!projected)
        return fail(projected.error());
    auto& [gate, up] = *projected;
    auto activated = silu_mul(context, gate, up);
    if (!activated)
        return fail(activated.error());
    return linear_add(context, *activated, down, hidden_states);
}

result<metal_tensor, tensor_op_error>
embed_tokens(const metal_context& context,
             const metal_tensor& weight,
             std::span<const token_id> tokens)
{
    if (tokens.empty()) {
        return fail(tensor_op_errc::empty_tokens);
    }

    const auto hidden_size = weight.descriptor().shape().dimensions()[1];
    auto token_ids = allocate_tensor(context, dtype::i32, { tokens.size() });
    if (!token_ids) {
        return fail(token_ids.error());
    }
    auto output = allocate_tensor(context, dtype::f32, { tokens.size(), hidden_size });
    if (!output) {
        return fail(output.error());
    }

    std::memcpy(token_ids->buffer().bytes().data(), tokens.data(), tokens.size_bytes());
    auto embedded = embedding_lookup(context, *token_ids, weight, *output);
    if (!embedded) {
        return fail(embedded.error() == tensor_op_errc::token_out_of_range
                        ? tensor_op_errc::token_out_of_range
                        : tensor_op_errc::backend_failure);
    }
    return std::move(*output);
}

result<metal_tensor, tensor_op_error>
encode_greedy(const metal_context& context,
              const metal_tensor& norm_weight,
              const metal_tensor& vocabulary_weight,
              float epsilon,
              const metal_tensor& hidden_states,
              std::span<const std::size_t> logits_indices,
              bool zero_centered)
{
    const auto& norm_shape = norm_weight.descriptor().shape();
    const auto& vocabulary_shape = vocabulary_weight.descriptor().shape();
    if (norm_shape.rank() != 1
        || vocabulary_shape.rank() != 2
        || norm_weight.descriptor().type() != dtype::bf16
        || vocabulary_weight.descriptor().type() != dtype::bf16
        || vocabulary_shape.dimensions()[1] != norm_shape.dimensions()[0]
        || !std::isfinite(epsilon)
        || epsilon <= 0) {
        return fail(tensor_op_errc::invalid_hidden_states);
    }
    const auto hidden_size = norm_shape.dimensions()[0];
    const auto vocabulary_size = vocabulary_shape.dimensions()[0];
    if (logits_indices.empty()) {
        return fail(tensor_op_errc::empty_logits_indices);
    }
    const auto& hidden_shape = hidden_states.descriptor().shape();
    if (hidden_shape.rank() != 2
        || hidden_states.descriptor().type() != dtype::f32
        || hidden_shape.dimensions()[1] != hidden_size) {
        return fail(tensor_op_errc::invalid_hidden_states);
    }
    if (vocabulary_size > static_cast<std::size_t>(std::numeric_limits<token_id>::max())) {
        return fail(tensor_op_errc::token_id_overflow);
    }

    const auto hidden_row_count = hidden_shape.dimensions()[0];
    auto row_indices = allocate_tensor(context, dtype::i32, { logits_indices.size() });
    if (!row_indices) {
        return fail(row_indices.error());
    }
    auto index_bytes = row_indices->buffer().bytes();
    for (std::size_t row = 0; row < logits_indices.size(); ++row) {
        const auto index = logits_indices[row];
        if (index >= hidden_row_count || index > std::numeric_limits<std::uint32_t>::max()) {
            return fail(tensor_op_errc::logits_index_out_of_range);
        }
        const auto kernel_index = static_cast<std::uint32_t>(index);
        std::memcpy(index_bytes.data() + row * sizeof(kernel_index), &kernel_index,
                    sizeof(kernel_index));
    }

    auto normalized = allocate_tensor(context, dtype::f32, { logits_indices.size(), hidden_size });
    if (!normalized) {
        return fail(normalized.error());
    }
    const auto partial_count =
        (vocabulary_size + metal_kernels::greedy_argmax_outputs_per_threadgroup - 1)
        / metal_kernels::greedy_argmax_outputs_per_threadgroup;
    auto partial_maxima =
        allocate_tensor(context, dtype::f32, { logits_indices.size(), partial_count, 2 });
    if (!partial_maxima) {
        return fail(partial_maxima.error());
    }
    auto token_ids = allocate_tensor(context, dtype::i32, { logits_indices.size() });
    if (!token_ids) {
        return fail(token_ids.error());
    }

    const auto operation = metal_kernels(context).dispatch_greedy_vocabulary_bf16(
        hidden_states.buffer(), row_indices->buffer(), norm_weight.buffer(),
        vocabulary_weight.buffer(), normalized->buffer(), partial_maxima->buffer(),
        token_ids->buffer(), logits_indices.size(), hidden_size, vocabulary_size, partial_count,
        epsilon, zero_centered);
    if (!operation) {
        return fail(tensor_op_errc::backend_failure, operation.error(), "greedy projection");
    }
    return std::move(*token_ids);
}

std::vector<token_id>
read_greedy(const metal_tensor& token_ids)
{
    const auto& shape = token_ids.descriptor().shape();
    std::vector<token_id> tokens(shape.dimensions()[0]);
    std::memcpy(tokens.data(), token_ids.buffer().bytes().data(), tokens.size() * sizeof(token_id));
    return tokens;
}

} // namespace chibillm
