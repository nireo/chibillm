#include "tensor/layers.h"
#include "tensor/tensor_ops.h"

namespace chibillm {
result<metal_tensor, tensor_op_errc>
normalized_swiglu(const metal_context& context,
                  const metal_tensor& norm,
                  const metal_tensor& gateup,
                  const metal_tensor& down,
                  float epsilon,
                  const metal_tensor& hidden_states)
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

    const auto rows = shape.dimensions()[0];
    auto normalized = allocate_tensor(context, dtype::f32, { rows, hidden_size });
    if (!normalized)
        return fail(normalized.error());
    CL_TRY(rms_norm(context, hidden_states, norm, epsilon, *normalized));
    auto gate = allocate_tensor(context, dtype::f32, { rows, intermediate_size });
    if (!gate)
        return fail(gate.error());
    auto up = allocate_tensor(context, dtype::f32, { rows, intermediate_size });
    if (!up)
        return fail(up.error());
    CL_TRY(linear_split(context, *normalized, gateup, { &*gate, &*up }));
    auto activated = allocate_tensor(context, dtype::f32, { rows, intermediate_size });
    if (!activated)
        return fail(activated.error());
    CL_TRY(silu_mul(context, *gate, *up, *activated));
    auto residual = allocate_tensor(context, dtype::f32, { rows, hidden_size });
    if (!residual)
        return fail(residual.error());
    CL_TRY(linear_add(context, *activated, down, hidden_states, *residual));

    return std::move(*residual);
}

} // namespace chibillm
