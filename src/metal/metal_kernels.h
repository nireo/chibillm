#pragma once
#include "metal/metal_context.h"
#include <array>

namespace chibillm {
class metal_kernels {
public:
    explicit metal_kernels(const metal_context& context)
        : context_(context)
    {}

    static constexpr std::size_t greedy_argmax_outputs_per_threadgroup = 64;
    [[nodiscard]] result<void, metal_error>
    dispatch_linear_add_bf16(const metal_buffer& input,
                             const metal_buffer& weight,
                             const metal_buffer& residual,
                             metal_buffer& output,
                             std::size_t input_features,
                             std::size_t output_features) const;

    [[nodiscard]] result<void, metal_error>
    dispatch_linear_split_bf16(const metal_buffer& input,
                               const metal_buffer& weight,
                               const std::array<metal_buffer*, 3>& outputs,
                               std::size_t rows,
                               std::size_t input_features,
                               const std::array<std::size_t, 3>& widths) const;

    [[nodiscard]] result<void, metal_error> dispatch_embedding_bf16(const metal_buffer& token_ids,
                                                                    const metal_buffer& weight,
                                                                    metal_buffer& output,
                                                                    std::size_t token_count,
                                                                    std::size_t hidden_size) const;

    [[nodiscard]] result<void, metal_error> dispatch_rms_norm_bf16(const metal_buffer& input,
                                                                   const metal_buffer& weight,
                                                                   metal_buffer& output,
                                                                   std::size_t rows,
                                                                   std::size_t hidden_size,
                                                                   float epsilon) const;

    [[nodiscard]] result<void, metal_error>
    dispatch_greedy_vocabulary_bf16(const metal_buffer& hidden_states,
                                    const metal_buffer& row_indices,
                                    const metal_buffer& norm_weight,
                                    const metal_buffer& vocabulary_weight,
                                    metal_buffer& normalized,
                                    metal_buffer& partial_maxima,
                                    metal_buffer& token_ids,
                                    std::size_t rows,
                                    std::size_t hidden_size,
                                    std::size_t vocabulary_size,
                                    std::size_t partial_count,
                                    float epsilon) const;

    [[nodiscard]] result<void, metal_error> dispatch_silu_mul_f32(const metal_buffer& gate,
                                                                  const metal_buffer& up,
                                                                  metal_buffer& output,
                                                                  std::size_t element_count) const;

    [[nodiscard]] result<void, metal_error> dispatch_add_f32(const metal_buffer& lhs,
                                                             const metal_buffer& rhs,
                                                             metal_buffer& output,
                                                             std::size_t element_count) const;

    [[nodiscard]] result<void, metal_error> dispatch_rope_f32(const metal_buffer& input,
                                                              const metal_buffer& positions,
                                                              metal_buffer& output,
                                                              std::size_t rows,
                                                              std::size_t head_count,
                                                              std::size_t head_dimension,
                                                              float theta) const;

    [[nodiscard]] result<void, metal_error> dispatch_store_kv_f32(const metal_buffer& keys,
                                                                  const metal_buffer& values,
                                                                  const metal_buffer& slot_mapping,
                                                                  metal_buffer& key_cache,
                                                                  metal_buffer& value_cache,
                                                                  std::size_t rows,
                                                                  std::size_t feature_count,
                                                                  std::size_t layer,
                                                                  std::size_t slot_count) const;

    [[nodiscard]] result<void, metal_error>
    dispatch_paged_attention_f32(const metal_buffer& queries,
                                 const metal_buffer& positions,
                                 const metal_buffer& block_table,
                                 const metal_buffer& block_table_offsets,
                                 const metal_buffer& block_table_lengths,
                                 const metal_buffer& key_cache,
                                 const metal_buffer& value_cache,
                                 metal_buffer& output,
                                 std::size_t rows,
                                 std::size_t query_head_count,
                                 std::size_t kv_head_count,
                                 std::size_t head_dimension,
                                 std::size_t block_size,
                                 std::size_t slot_count,
                                 std::size_t layer,
                                 std::size_t block_table_entry_count) const;

    [[nodiscard]] result<void, metal_error>
    dispatch_paged_flash_attention_prefill_f32(const metal_buffer& queries,
                                               const metal_buffer& positions,
                                               const metal_buffer& block_table,
                                               const metal_buffer& block_table_offsets,
                                               const metal_buffer& block_table_lengths,
                                               const metal_buffer& key_cache,
                                               const metal_buffer& value_cache,
                                               const metal_buffer& query_tile_starts,
                                               const metal_buffer& query_tile_lengths,
                                               metal_buffer& output,
                                               std::size_t rows,
                                               std::size_t query_head_count,
                                               std::size_t kv_head_count,
                                               std::size_t head_dimension,
                                               std::size_t block_size,
                                               std::size_t slot_count,
                                               std::size_t layer,
                                               std::size_t block_table_entry_count,
                                               std::size_t query_tile_count) const;

private:
    const metal_context& context_;
};
} // namespace chibillm
