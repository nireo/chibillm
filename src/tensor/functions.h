#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "seq.h"
#include "tensor/matrix.h"
#include "tensor/tensor_ops.h"

namespace chibillm {

class prepared_attention_batch;

// These helpers allocate owned f32 outputs. Explicit output-buffer operations
// remain available in tensor_ops.h, attention.h, and deltanet.h for in-place
// execution and reuse. Both forms enqueue work without adding synchronization.
[[nodiscard]] result<metal_tensor, tensor_op_error>
linear(const metal_context& context, const metal_tensor& input, matrix_view weight);
[[nodiscard]] result<metal_tensor, tensor_op_error> linear_add(const metal_context& context,
                                                               const metal_tensor& input,
                                                               matrix_view weight,
                                                               const metal_tensor& residual);
[[nodiscard]] result<std::array<metal_tensor, 2>, tensor_op_error>
linear_split(const metal_context& context,
             const metal_tensor& input,
             matrix_view packed_weight,
             std::size_t first_width,
             std::size_t second_width);
[[nodiscard]] result<std::array<metal_tensor, 3>, tensor_op_error>
linear_split(const metal_context& context,
             const metal_tensor& input,
             matrix_view packed_weight,
             std::size_t first_width,
             std::size_t second_width,
             std::size_t third_width);
[[nodiscard]] result<metal_tensor, tensor_op_error> rms_norm(const metal_context& context,
                                                             const metal_tensor& input,
                                                             const metal_tensor& weight,
                                                             float epsilon,
                                                             bool zero_centered = false);
[[nodiscard]] result<metal_tensor, tensor_op_error>
silu_mul(const metal_context& context, const metal_tensor& gate, const metal_tensor& up);
[[nodiscard]] result<std::array<metal_tensor, 2>, tensor_op_error>
split_heads(const metal_context& context, const metal_tensor& input, std::size_t head_count);
[[nodiscard]] result<metal_tensor, tensor_op_error> rope(const metal_context& context,
                                                         const metal_tensor& input,
                                                         const metal_tensor& positions,
                                                         std::size_t head_count,
                                                         float theta,
                                                         std::size_t rotary_dimension = 0);

[[nodiscard]] result<metal_tensor, tensor_op_error>
paged_attention(const metal_context& context,
                const metal_tensor& queries,
                const prepared_attention_batch& metadata,
                std::size_t layer,
                std::size_t query_head_count,
                const metal_kv_cache& cache);

// Normalizes each head and applies weight and SiLU(gate).
[[nodiscard]] result<metal_tensor, tensor_op_error> rms_norm_gated(const metal_context& context,
                                                                   const metal_tensor& input,
                                                                   const metal_tensor& gate,
                                                                   const metal_tensor& weight,
                                                                   float epsilon);

result<metal_tensor, tensor_op_error> normalized_swiglu(const metal_context& context,
                                                        const metal_tensor& norm,
                                                        matrix_view gateup,
                                                        matrix_view down,
                                                        float epsilon,
                                                        const metal_tensor& hidden_states,
                                                        bool zero_centered = false);

[[nodiscard]] result<metal_tensor, tensor_op_error>
embed_tokens(const metal_context& context, matrix_view weight, std::span<const token_id> tokens);

// Encodes row gather, final norm, vocabulary projection, and argmax into the
// current Metal command buffer. Only the small token-ID tensor is returned.
[[nodiscard]] result<metal_tensor, tensor_op_error>
encode_greedy(const metal_context& context,
              const metal_tensor& norm_weight,
              matrix_view vocabulary_weight,
              float epsilon,
              const metal_tensor& hidden_states,
              std::span<const std::size_t> logits_indices,
              bool zero_centered = false);

// Reads token IDs after the command buffer containing encode_greedy has completed.
[[nodiscard]] std::vector<token_id> read_greedy(const metal_tensor& token_ids);

} // namespace chibillm
