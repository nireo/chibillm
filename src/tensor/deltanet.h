#pragma once

#include "tensor/tensor_ops.h"

namespace chibillm {

// One sequence per call; rows are consecutive tokens. State must be initialized
// to zero for a new sequence and preserved across chunks. Calls in a compute
// pass are asynchronous until the pass finishes. The caller owns rollback of
// mutated state on failure. Outputs and state must not alias any other argument.
//
// input/output: f32 [tokens, channels], weight: bf16 [channels, 1, kernel].
// history: f32 [channels, kernel], oldest to newest raw (pre-SiLU) inputs.
[[nodiscard]] result<void, tensor_op_errc> causal_conv1d_silu(const metal_context& context,
                                                              const metal_tensor& input,
                                                              const metal_tensor& weight,
                                                              metal_tensor& history,
                                                              metal_tensor& output);

// Fused Qwen3.5 gate preparation and recurrent gated delta rule.
// qkv: f32 [tokens, 2 * key_heads * key_dim + value_heads * value_dim],
// packed Q | K | V after convolution. Q/K are L2-normalized with epsilon;
// Q is also scaled by 1/sqrt(key_dim). Value heads repeat consecutive Q/K
// heads (value_heads must be divisible by key_heads).
// a/b: f32 [tokens, value_heads]; A_log: f32 [value_heads];
// dt_bias: bf16 [value_heads]. beta = sigmoid(b),
// g = -exp(A_log) * softplus(a + dt_bias).
// state: f32 [value_heads, key_dim, value_dim], updated in place by
// S *= exp(g); S += k outer (beta * (v - k^T S)); output = q^T S.
// output: f32 [tokens, value_heads * value_dim]. A sequential scan supports
// decode and arbitrary prefill chunks; this is not a parallel prefill kernel.
[[nodiscard]] result<void, tensor_op_errc> gated_delta_rule(const metal_context& context,
                                                            const metal_tensor& qkv,
                                                            const metal_tensor& a,
                                                            const metal_tensor& b,
                                                            const metal_tensor& A_log,
                                                            const metal_tensor& dt_bias,
                                                            std::size_t key_heads,
                                                            metal_tensor& state,
                                                            metal_tensor& output,
                                                            float epsilon = 1e-6F);

// input/gate/output: f32 [tokens, heads * value_dim], weight: f32 [value_dim].
// Each head is normalized independently, then scaled by weight and SiLU(gate).
// This norm uses weight directly, NOT the zero-centered (1 + weight) variant.
[[nodiscard]] result<void, tensor_op_errc> rms_norm_gated(const metal_context& context,
                                                          const metal_tensor& input,
                                                          const metal_tensor& gate,
                                                          const metal_tensor& weight,
                                                          float epsilon,
                                                          metal_tensor& output);

} // namespace chibillm
