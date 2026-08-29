#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "metal/metal_context.h"
#include "metal/metal_tensor.h"
#include "qwen/qwen_config.h"
#include "qwen/qwen_weights.h"
#include "result.h"
#include "seq.h"

namespace chibillm {

enum class qwen_output_errc : std::uint8_t {
    empty_logits_indices,
    invalid_hidden_states,
    logits_index_out_of_range,
    token_id_overflow,
    tensor_creation_failed,
    metal_allocation_failed,
    backend_failure,
};

// Encodes row gather, final norm, vocabulary projection, and argmax into the
// current Metal command buffer. Only the small token-ID tensor is returned.
[[nodiscard]] result<metal_tensor, qwen_output_errc>
encode_qwen_greedy(const metal_context& context,
                   const qwen_config& config,
                   const qwen_weights& weights,
                   const metal_tensor& hidden_states,
                   std::span<const std::size_t> logits_indices);

// Reads token IDs after the command buffer containing encode_qwen_greedy has completed.
[[nodiscard]] std::vector<token_id> read_qwen_greedy(const metal_tensor& token_ids);

// Standalone convenience wrapper. When no compute pass is open, encoding waits
// for its one command buffer and this returns its token IDs immediately.
[[nodiscard]] result<std::vector<token_id>, qwen_output_errc>
sample_qwen_greedy(const metal_context& context,
                   const qwen_config& config,
                   const qwen_weights& weights,
                   const metal_tensor& hidden_states,
                   std::span<const std::size_t> logits_indices);

} // namespace chibillm
