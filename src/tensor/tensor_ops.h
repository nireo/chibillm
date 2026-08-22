#pragma once

#include <cstdint>

#include "metal/metal_context.h"
#include "metal/metal_tensor.h"
#include "result.h"

namespace chibillm {

enum class tensor_op_errc : std::uint8_t {
    invalid_rank,
    unsupported_dtype,
    inner_dimension_mismatch,
    output_shape_mismatch,
    token_out_of_range,
    invalid_epsilon,
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

} // namespace chibillm
