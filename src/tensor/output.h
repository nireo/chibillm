#include "tensor/tensor_ops.h"
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "metal/metal_context.h"
#include "metal/metal_tensor.h"
#include "result.h"
#include "seq.h"

namespace chibillm {

// Encodes row gather, final norm, vocabulary projection, and argmax into the
// current Metal command buffer. Only the small token-ID tensor is returned.
[[nodiscard]] result<metal_tensor, tensor_op_errc>
encode_greedy(const metal_context& context,
              const metal_tensor& norm_weight,
              const metal_tensor& vocabulary_weight,
              float epsilon,
              const metal_tensor& hidden_states,
              std::span<const std::size_t> logits_indices);

// Reads token IDs after the command buffer containing encode_greedy has completed.
[[nodiscard]] std::vector<token_id> read_greedy(const metal_tensor& token_ids);

} // namespace chibillm
