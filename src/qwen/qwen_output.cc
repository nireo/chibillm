#include "qwen/qwen_output.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "tensor/dtype.h"
#include "tensor/tensor_ops.h"

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

result<std::vector<token_id>, qwen_output_errc>
sample_qwen_greedy(const metal_context& context,
                   const qwen_config& config,
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
    for (const auto index : logits_indices) {
        if (index >= hidden_row_count) {
            return fail(qwen_output_errc::logits_index_out_of_range);
        }
    }

    auto selected =
        metal_tensor::make(context, dtype::f32, { logits_indices.size(), config.hidden_size });
    if (!selected) {
        return fail(tensor_error(selected.error()));
    }
    const auto row_size = config.hidden_size * sizeof(float);
    const auto hidden_bytes = hidden_states.buffer().bytes();
    auto selected_bytes = selected->buffer().bytes();
    for (std::size_t row = 0; row < logits_indices.size(); ++row) {
        std::memcpy(selected_bytes.data() + row * row_size,
                    hidden_bytes.data() + logits_indices[row] * row_size, row_size);
    }

    auto normalized =
        metal_tensor::make(context, dtype::f32, { logits_indices.size(), config.hidden_size });
    if (!normalized) {
        return fail(tensor_error(normalized.error()));
    }
    auto operation =
        rms_norm(context, *selected, weights.final_norm, config.rms_epsilon, *normalized);
    if (!operation) {
        return fail(qwen_output_errc::backend_failure);
    }

    auto logits =
        metal_tensor::make(context, dtype::f32, { logits_indices.size(), config.vocabulary_size });
    if (!logits) {
        return fail(tensor_error(logits.error()));
    }
    operation = linear(context, *normalized, weights.output, *logits);
    if (!operation) {
        return fail(qwen_output_errc::backend_failure);
    }

    const auto logits_bytes = logits->buffer().bytes();
    const auto* values = reinterpret_cast<const float*>(logits_bytes.data());
    std::vector<token_id> tokens;
    tokens.reserve(logits_indices.size());
    for (std::size_t row = 0; row < logits_indices.size(); ++row) {
        const auto* begin = values + row * config.vocabulary_size;
        const auto* best = std::max_element(begin, begin + config.vocabulary_size);
        tokens.push_back(static_cast<token_id>(best - begin));
    }
    return tokens;
}

} // namespace chibillm
