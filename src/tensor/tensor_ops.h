#pragma once

#include <cstddef>
#include <cstdint>

#include "metal/metal_context.h"
#include "metal/metal_tensor.h"
#include "result.h"

namespace chibillm {

enum class tensor_op_errc : std::uint8_t {
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
    backend_failure,
};

// computes lhs [m, k] times rhs [k, n] into output [m, n].
[[nodiscard]] result<void, tensor_op_errc> matmul(const metal_context& context,
                                                  const metal_tensor& lhs,
                                                  const metal_tensor& rhs,
                                                  metal_tensor& output);

// projects input [m, k] with bf16 weight [n, k] into f32 output [m, n].
[[nodiscard]] result<void, tensor_op_errc> linear(const metal_context& context,
                                                  const metal_tensor& input,
                                                  const metal_tensor& weight,
                                                  metal_tensor& output);

// gathers i32 token ids [t] from bf16 weight [vocabulary, hidden] into f32 output [t, hidden].
[[nodiscard]] result<void, tensor_op_errc> embedding_lookup(const metal_context& context,
                                                            const metal_tensor& token_ids,
                                                            const metal_tensor& weight,
                                                            metal_tensor& output);

// normalizes each f32 input row and scales it with a bf16 hidden-size weight.
[[nodiscard]] result<void, tensor_op_errc> rms_norm(const metal_context& context,
                                                    const metal_tensor& input,
                                                    const metal_tensor& weight,
                                                    float epsilon,
                                                    metal_tensor& output);

// applies silu to gate and multiplies it elementwise by up.
[[nodiscard]] result<void, tensor_op_errc> silu_mul(const metal_context& context,
                                                    const metal_tensor& gate,
                                                    const metal_tensor& up,
                                                    metal_tensor& output);

// adds lhs and rhs elementwise into output.
[[nodiscard]] result<void, tensor_op_errc> add(const metal_context& context,
                                               const metal_tensor& lhs,
                                               const metal_tensor& rhs,
                                               metal_tensor& output);

// applies qwen rotary positions to flattened attention heads.
[[nodiscard]] result<void, tensor_op_errc> rope(const metal_context& context,
                                                const metal_tensor& input,
                                                const metal_tensor& positions,
                                                std::size_t head_count,
                                                float theta,
                                                metal_tensor& output);

} // namespace chibillm
