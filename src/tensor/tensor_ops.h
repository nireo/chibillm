#pragma once

#include "error.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "metal/metal_context.h"
#include "metal/metal_kv_cache.h"
#include "metal/metal_tensor.h"
#include "result.h"

namespace chibillm {

enum class tensor_op_errc : std::uint8_t {
    tensor_creation_failed,
    allocation_failed,
    empty_tokens,
    empty_logits_indices,
    invalid_hidden_states,
    logits_index_out_of_range,
    token_id_overflow,
    invalid_rank,
    unsupported_dtype,
    input_shape_mismatch,
    inner_dimension_mismatch,
    output_shape_mismatch,
    token_out_of_range,
    invalid_epsilon,
    position_count_mismatch,
    invalid_head_count,
    invalid_head_dimension,
    invalid_rope_theta,
    cache_slot_count_mismatch,
    cache_feature_count_mismatch,
    cache_layer_out_of_range,
    cache_slot_out_of_range,
    block_table_metadata_count_mismatch,
    block_table_range_out_of_bounds,
    cache_block_out_of_range,
    cache_head_dimension_mismatch,
    invalid_kv_head_mapping,
    backend_failure,
    unsupported_aliasing,
};

[[nodiscard]] inline std::string_view
error_name(tensor_op_errc code) noexcept
{
    static constexpr std::array names {
        "tensor_op.tensor_creation_failed",
        "tensor_op.allocation_failed",
        "tensor_op.empty_tokens",
        "tensor_op.empty_logits_indices",
        "tensor_op.invalid_hidden_states",
        "tensor_op.logits_index_out_of_range",
        "tensor_op.token_id_overflow",
        "tensor_op.invalid_rank",
        "tensor_op.unsupported_dtype",
        "tensor_op.input_shape_mismatch",
        "tensor_op.inner_dimension_mismatch",
        "tensor_op.output_shape_mismatch",
        "tensor_op.token_out_of_range",
        "tensor_op.invalid_epsilon",
        "tensor_op.position_count_mismatch",
        "tensor_op.invalid_head_count",
        "tensor_op.invalid_head_dimension",
        "tensor_op.invalid_rope_theta",
        "tensor_op.cache_slot_count_mismatch",
        "tensor_op.cache_feature_count_mismatch",
        "tensor_op.cache_layer_out_of_range",
        "tensor_op.cache_slot_out_of_range",
        "tensor_op.block_table_metadata_count_mismatch",
        "tensor_op.block_table_range_out_of_bounds",
        "tensor_op.cache_block_out_of_range",
        "tensor_op.cache_head_dimension_mismatch",
        "tensor_op.invalid_kv_head_mapping",
        "tensor_op.backend_failure",
        "tensor_op.unsupported_aliasing",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "tensor_op.unknown_error";
}

using tensor_op_error = error<tensor_op_errc>;

[[nodiscard]] result<metal_tensor, tensor_op_error>
allocate_tensor(const metal_context& context, dtype type, std::vector<std::size_t> dimensions);
[[nodiscard]] result<metal_tensor, tensor_op_error>
upload_u32(const metal_context& context, std::span<const std::uint32_t> values);

// projects input and adds a same-shaped f32 residual into output.
[[nodiscard]] result<void, tensor_op_error> linear_add(const metal_context& context,
                                                       const metal_tensor& input,
                                                       const metal_tensor& weight,
                                                       const metal_tensor& residual,
                                                       metal_tensor& output);

// Projects one input through one to three vertically packed bf16 weights in one launch.
[[nodiscard]] result<void, tensor_op_error>
linear_split(const metal_context& context,
             const metal_tensor& input,
             const metal_tensor& packed_weight,
             std::initializer_list<metal_tensor*> outputs);

// gathers i32 token ids [t] from bf16 weight [vocabulary, hidden] into f32 output [t, hidden].
[[nodiscard]] result<void, tensor_op_error> embedding_lookup(const metal_context& context,
                                                             const metal_tensor& token_ids,
                                                             const metal_tensor& weight,
                                                             metal_tensor& output);

// Normalizes each contiguous weight-sized group. Zero-centered norms use
// (1 + weight) in FP32; ordinary norms use the bf16 weight directly.
[[nodiscard]] result<void, tensor_op_error> rms_norm(const metal_context& context,
                                                     const metal_tensor& input,
                                                     const metal_tensor& weight,
                                                     float epsilon,
                                                     metal_tensor& output,
                                                     bool zero_centered = false);

// applies silu to gate and multiplies it elementwise by up.
[[nodiscard]] result<void, tensor_op_error> silu_mul(const metal_context& context,
                                                     const metal_tensor& gate,
                                                     const metal_tensor& up,
                                                     metal_tensor& output);

// Multiplies input by sigmoid(gate), without the extra gate factor in SiLU.
[[nodiscard]] result<void, tensor_op_error> sigmoid_mul(const metal_context& context,
                                                        const metal_tensor& gate,
                                                        const metal_tensor& input,
                                                        metal_tensor& output);

// Splits [rows, heads * 2 * head_dim] into two [rows, heads * head_dim] tensors.
// Each head stores its first half followed by its second half; outputs cannot alias.
[[nodiscard]] result<void, tensor_op_error> split_heads(const metal_context& context,
                                                        const metal_tensor& input,
                                                        std::size_t head_count,
                                                        metal_tensor& first,
                                                        metal_tensor& second);

// adds lhs and rhs elementwise into output.
[[nodiscard]] result<void, tensor_op_error> add(const metal_context& context,
                                                const metal_tensor& lhs,
                                                const metal_tensor& rhs,
                                                metal_tensor& output);

// Rotates the first rotary_dimension features of each head, preserving the tail.
// Zero selects the full head. Text-only Qwen3.5 uses the same position on all MRoPE axes.
[[nodiscard]] result<void, tensor_op_error> rope(const metal_context& context,
                                                 const metal_tensor& input,
                                                 const metal_tensor& positions,
                                                 std::size_t head_count,
                                                 float theta,
                                                 metal_tensor& output,
                                                 std::size_t rotary_dimension = 0);

// writes new f32 keys and values into physical cache slots for one layer.
[[nodiscard]] result<void, tensor_op_error> store_kv(const metal_context& context,
                                                     const metal_tensor& keys,
                                                     const metal_tensor& values,
                                                     const metal_tensor& slot_mapping,
                                                     std::size_t layer,
                                                     metal_kv_cache& cache);

// attends each query row through its paged block table for one layer.
[[nodiscard]] result<void, tensor_op_error> paged_attention(const metal_context& context,
                                                            const metal_tensor& queries,
                                                            const metal_tensor& positions,
                                                            const metal_tensor& block_table,
                                                            const metal_tensor& block_table_offsets,
                                                            const metal_tensor& block_table_lengths,
                                                            std::size_t layer,
                                                            std::size_t query_head_count,
                                                            const metal_kv_cache& cache,
                                                            metal_tensor& output);

} // namespace chibillm
