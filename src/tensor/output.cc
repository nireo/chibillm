#include "tensor/output.h"
#include "metal/metal_kernels.h"
#include "tensor/tensor_ops.h"
#include <cmath>

#include <cstring>
#include <limits>
#include <vector>

#include "tensor/dtype.h"

namespace chibillm {

result<metal_tensor, tensor_op_errc>
encode_greedy(const metal_context& context,
              const metal_tensor& norm_weight,
              const metal_tensor& vocabulary_weight,
              float epsilon,
              const metal_tensor& hidden_states,
              std::span<const std::size_t> logits_indices)
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
        const auto shader_index = static_cast<std::uint32_t>(index);
        std::memcpy(index_bytes.data() + row * sizeof(shader_index), &shader_index,
                    sizeof(shader_index));
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
        epsilon);
    if (!operation) {
        return fail(tensor_op_errc::backend_failure);
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
