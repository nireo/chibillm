#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string_view>

#include "metal/metal_buffer.h"
#include "metal/metal_error.h"
#include "result.h"

namespace chibillm {

class metal_tensor;
class metal_kv_cache;
struct qwen_config;
struct qwen_weights;
enum class qwen_output_errc : std::uint8_t;
enum class tensor_op_errc : std::uint8_t;

// owns the metal device, queue, shader library, and compute pipelines.
class metal_context {
public:
    [[nodiscard]] static result<metal_context, metal_error> make(std::string_view shader_source);

    metal_context(const metal_context&) = delete;
    metal_context& operator=(const metal_context&) = delete;
    metal_context(metal_context&&) noexcept;
    metal_context& operator=(metal_context&&) noexcept;
    ~metal_context();

    [[nodiscard]] std::string_view device_name() const noexcept;

    // opens one command buffer whose compute encoder collects every dispatch until
    // end_compute_pass(). Dispatches made while a pass is open are appended to it;
    // they no longer wait on the GPU individually.
    [[nodiscard]] result<void, metal_error> begin_compute_pass();

    // commits the open command buffer and waits once for all of its kernels.
    [[nodiscard]] result<void, metal_error> end_compute_pass();

    // commits (and waits on) whatever kernels were already encoded into the open
    // pass, then closes it. Used when a batched operation fails midway so GPU and
    // shared-memory state still settle before returning to the caller.
    void abort_compute_pass() noexcept;

    [[nodiscard]] result<metal_buffer, metal_error>
    make_shared_buffer(std::size_t size_bytes) const;

private:
    friend result<void, tensor_op_errc> matmul(const metal_context& context,
                                               const metal_tensor& lhs,
                                               const metal_tensor& rhs,
                                               metal_tensor& output);

    friend result<void, tensor_op_errc> linear(const metal_context& context,
                                               const metal_tensor& input,
                                               const metal_tensor& weight,
                                               metal_tensor& output);

    friend result<void, tensor_op_errc> linear_add(const metal_context& context,
                                                   const metal_tensor& input,
                                                   const metal_tensor& weight,
                                                   const metal_tensor& residual,
                                                   metal_tensor& output);

    friend result<void, tensor_op_errc> linear_split(const metal_context& context,
                                                     const metal_tensor& input,
                                                     const metal_tensor& weight,
                                                     std::initializer_list<metal_tensor*> outputs);

    friend result<void, tensor_op_errc> embedding_lookup(const metal_context& context,
                                                         const metal_tensor& token_ids,
                                                         const metal_tensor& weight,
                                                         metal_tensor& output);

    friend result<void, tensor_op_errc> rms_norm(const metal_context& context,
                                                 const metal_tensor& input,
                                                 const metal_tensor& weight,
                                                 float epsilon,
                                                 metal_tensor& output);

    friend result<void, tensor_op_errc> rms_norm_heads(const metal_context& context,
                                                       const metal_tensor& input,
                                                       const metal_tensor& weight,
                                                       float epsilon,
                                                       metal_tensor& output);

    friend result<metal_tensor, qwen_output_errc>
    encode_qwen_greedy(const metal_context& context,
                       const qwen_config& config,
                       const qwen_weights& weights,
                       const metal_tensor& hidden_states,
                       std::span<const std::size_t> logits_indices);

    friend result<void, tensor_op_errc> silu_mul(const metal_context& context,
                                                 const metal_tensor& gate,
                                                 const metal_tensor& up,
                                                 metal_tensor& output);

    friend result<void, tensor_op_errc> add(const metal_context& context,
                                            const metal_tensor& lhs,
                                            const metal_tensor& rhs,
                                            metal_tensor& output);

    friend result<void, tensor_op_errc> rope(const metal_context& context,
                                             const metal_tensor& input,
                                             const metal_tensor& positions,
                                             std::size_t head_count,
                                             float theta,
                                             metal_tensor& output);

    friend result<void, tensor_op_errc> store_kv(const metal_context& context,
                                                 const metal_tensor& keys,
                                                 const metal_tensor& values,
                                                 const metal_tensor& slot_mapping,
                                                 std::size_t layer,
                                                 metal_kv_cache& cache);

    friend result<void, tensor_op_errc> paged_attention(const metal_context& context,
                                                        const metal_tensor& queries,
                                                        const metal_tensor& positions,
                                                        const metal_tensor& block_table,
                                                        const metal_tensor& block_table_offsets,
                                                        const metal_tensor& block_table_lengths,
                                                        std::size_t layer,
                                                        std::size_t query_head_count,
                                                        const metal_kv_cache& cache,
                                                        metal_tensor& output);

    struct implementation;

    static constexpr std::size_t greedy_argmax_outputs_per_threadgroup = 32;

    explicit metal_context(std::unique_ptr<implementation> implementation) noexcept;

    [[nodiscard]] result<void, metal_error> dispatch_matmul(const metal_buffer& lhs,
                                                            const metal_buffer& rhs,
                                                            metal_buffer& output,
                                                            std::size_t rows,
                                                            std::size_t inner_dimension,
                                                            std::size_t columns) const;

    [[nodiscard]] result<void, metal_error> dispatch_linear_bf16(const metal_buffer& input,
                                                                 const metal_buffer& weight,
                                                                 metal_buffer& output,
                                                                 std::size_t rows,
                                                                 std::size_t input_features,
                                                                 std::size_t output_features) const;

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

    std::unique_ptr<implementation> implementation_;
};

} // namespace chibillm
