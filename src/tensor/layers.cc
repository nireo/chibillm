#include "tensor/layers.h"
#include "tensor/tensor_ops.h"

namespace chibillm {
result<metal_tensor, tensor_op_error>
normalized_swiglu(const metal_context& context,
                  const metal_tensor& norm,
                  const metal_tensor& gateup,
                  const metal_tensor& down,
                  float epsilon,
                  const metal_tensor& hidden_states,
                  bool zero_centered)
{
    if (norm.descriptor().shape().rank() != 1 || down.descriptor().shape().rank() != 2) {
        return fail(tensor_op_errc::input_shape_mismatch);
    }
    const auto hidden_size = norm.descriptor().shape().dimensions()[0];
    const auto intermediate_size = down.descriptor().shape().dimensions()[1];
    const auto& shape = hidden_states.descriptor().shape();
    if (shape.rank() != 2
        || hidden_states.descriptor().type() != dtype::f32
        || shape.dimensions()[1] != hidden_size) {
        return fail(tensor_op_errc::input_shape_mismatch);
    }

    auto normalized = rms_norm(context, hidden_states, norm, epsilon, zero_centered);
    if (!normalized)
        return fail(normalized.error());
    auto projected =
        linear_split(context, *normalized, gateup, intermediate_size, intermediate_size);
    if (!projected)
        return fail(projected.error());
    auto& [gate, up] = *projected;
    auto activated = silu_mul(context, gate, up);
    if (!activated)
        return fail(activated.error());
    return linear_add(context, *activated, down, hidden_states);
}

} // namespace chibillm
