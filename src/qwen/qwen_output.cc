#include "qwen/qwen_output.h"

#include <cstring>
#include <limits>
#include <vector>

#include "tensor/dtype.h"

namespace chibillm {
namespace {

qwen_output_errc
tensor_error(metal_tensor_errc error)
{
    return error == metal_tensor_errc::invalid_descriptor
        ? qwen_output_errc::tensor_creation_failed
        : qwen_output_errc::metal_allocation_failed;
}

} // namespace

result<metal_tensor, qwen_output_errc>
encode_qwen_greedy(const metal_context& context,
                   const qwen3_config& config,
                   const qwen_weights& weights,
                   const metal_tensor& hidden_states,
                   std::span<const std::size_t> logits_indices)
{
    if (logits_indices.empty()) {
        return fail(qwen_output_errc::empty_logits_indices);
    }
    const auto& hidden_shape = hidden_states.descriptor().shape();
    if (hidden_shape.rank() != 2
        || hidden_states.descriptor().type() != dtype::f32
        || hidden_shape.dimensions()[1] != config.hidden_size) {
        return fail(qwen_output_errc::invalid_hidden_states);
    }
    if (config.vocabulary_size > static_cast<std::size_t>(std::numeric_limits<token_id>::max())) {
        return fail(qwen_output_errc::token_id_overflow);
    }

    const auto hidden_row_count = hidden_shape.dimensions()[0];
    auto row_indices = metal_tensor::make(context, dtype::i32, { logits_indices.size() });
    if (!row_indices) {
        return fail(tensor_error(row_indices.error()));
    }
    auto index_bytes = row_indices->buffer().bytes();
    for (std::size_t row = 0; row < logits_indices.size(); ++row) {
        const auto index = logits_indices[row];
        if (index >= hidden_row_count || index > std::numeric_limits<std::uint32_t>::max()) {
            return fail(qwen_output_errc::logits_index_out_of_range);
        }
        const auto shader_index = static_cast<std::uint32_t>(index);
        std::memcpy(index_bytes.data() + row * sizeof(shader_index), &shader_index,
                    sizeof(shader_index));
    }

    auto normalized =
        metal_tensor::make(context, dtype::f32, { logits_indices.size(), config.hidden_size });
    if (!normalized) {
        return fail(tensor_error(normalized.error()));
    }
    const auto partial_count =
        (config.vocabulary_size + metal_context::greedy_argmax_outputs_per_threadgroup - 1)
        / metal_context::greedy_argmax_outputs_per_threadgroup;
    auto partial_maxima =
        metal_tensor::make(context, dtype::f32, { logits_indices.size(), partial_count, 2 });
    if (!partial_maxima) {
        return fail(tensor_error(partial_maxima.error()));
    }
    auto token_ids = metal_tensor::make(context, dtype::i32, { logits_indices.size() });
    if (!token_ids) {
        return fail(tensor_error(token_ids.error()));
    }

    const auto operation = context.dispatch_greedy_vocabulary_bf16(
        hidden_states.buffer(), row_indices->buffer(), weights.final_norm.buffer(),
        weights.output.buffer(), normalized->buffer(), partial_maxima->buffer(),
        token_ids->buffer(), logits_indices.size(), config.hidden_size, config.vocabulary_size,
        partial_count, config.rms_epsilon);
    if (!operation) {
        return fail(qwen_output_errc::backend_failure);
    }
    return std::move(*token_ids);
}

std::vector<token_id>
read_qwen_greedy(const metal_tensor& token_ids)
{
    const auto& shape = token_ids.descriptor().shape();
    std::vector<token_id> tokens(shape.dimensions()[0]);
    std::memcpy(tokens.data(), token_ids.buffer().bytes().data(), tokens.size() * sizeof(token_id));
    return tokens;
}

} // namespace chibillm
