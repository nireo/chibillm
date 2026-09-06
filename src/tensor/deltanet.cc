#include "tensor/deltanet.h"
#include "metal/metal_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace chibillm {
namespace {
    bool
    shape_is(const metal_tensor& tensor, std::initializer_list<std::size_t> dimensions)
    {
        const auto actual = tensor.descriptor().shape().dimensions();
        return std::ranges::equal(actual, dimensions);
    }

    bool
    aliases(const metal_tensor& writable, std::initializer_list<const metal_tensor*> tensors)
    {
        return std::ranges::any_of(
            tensors, [&](const auto* tensor) { return &writable.buffer() == &tensor->buffer(); });
    }

    bool
    all_f32(std::initializer_list<const metal_tensor*> tensors)
    {
        return std::ranges::all_of(
            tensors, [](const auto* tensor) { return tensor->descriptor().type() == dtype::f32; });
    }
} // namespace

result<void, tensor_op_errc>
causal_conv1d_silu(const metal_context& context,
    const metal_tensor& input,
    const metal_tensor& weight,
    metal_tensor& history,
    metal_tensor& output)
{
    if (input.descriptor().shape().rank() != 2
        || weight.descriptor().shape().rank() != 3
        || history.descriptor().shape().rank() != 2
        || output.descriptor().shape().rank() != 2)
        return fail(tensor_op_errc::invalid_rank);

    if (!all_f32({ &input, &history, &output }) || weight.descriptor().type() != dtype::bf16)
        return fail(tensor_op_errc::unsupported_dtype);

    const auto dims = input.descriptor().shape().dimensions();
    const auto kernel = weight.descriptor().shape().dimensions()[2];
    if (!shape_is(weight, { dims[1], 1, kernel }) || !shape_is(history, { dims[1], kernel }))
        return fail(tensor_op_errc::input_shape_mismatch);

    if (!shape_is(output, { dims[0], dims[1] }))
        return fail(tensor_op_errc::output_shape_mismatch);

    if (aliases(history, { &input, &weight, &output }) || aliases(output, { &input, &weight }))
        return fail(tensor_op_errc::unsupported_aliasing);

    if (!metal_kernels(context).dispatch_causal_conv1d_silu(input.buffer(), weight.buffer(),
            history.buffer(), output.buffer(),
            dims[0], dims[1], kernel))
        return fail(tensor_op_errc::backend_failure);

    return { };
}

result<void, tensor_op_errc>
gated_delta_rule(const metal_context& context,
    const metal_tensor& qkv,
    const metal_tensor& a,
    const metal_tensor& b,
    const metal_tensor& A_log,
    const metal_tensor& dt_bias,
    std::size_t key_heads,
    metal_tensor& state,
    metal_tensor& output,
    float epsilon)
{
    if (qkv.descriptor().shape().rank() != 2
        || a.descriptor().shape().rank() != 2
        || b.descriptor().shape().rank() != 2
        || A_log.descriptor().shape().rank() != 1
        || dt_bias.descriptor().shape().rank() != 1
        || state.descriptor().shape().rank() != 3
        || output.descriptor().shape().rank() != 2)
        return fail(tensor_op_errc::invalid_rank);

    if (!all_f32({ &qkv, &a, &b, &A_log, &state, &output })
        || dt_bias.descriptor().type() != dtype::bf16)
        return fail(tensor_op_errc::unsupported_dtype);

    if (!std::isfinite(epsilon) || epsilon <= 0)
        return fail(tensor_op_errc::invalid_epsilon);

    const auto dims = state.descriptor().shape().dimensions();
    const auto heads = dims[0], key_dim = dims[1], value_dim = dims[2];
    const auto rows = qkv.descriptor().shape().dimensions()[0];
    if (key_heads == 0 || heads % key_heads != 0)
        return fail(tensor_op_errc::invalid_head_count);

    // Products in state are already checked by tensor_shape; guard packed QKV addition.
    const auto value_width = heads * value_dim;
    if (key_heads > (std::numeric_limits<std::size_t>::max() - value_width) / key_dim / 2)
        return fail(tensor_op_errc::input_shape_mismatch);
    if (!shape_is(qkv, { rows, 2 * key_heads * key_dim + value_width })
        || !shape_is(a, { rows, heads })
        || !shape_is(b, { rows, heads })
        || !shape_is(A_log, { heads })
        || !shape_is(dt_bias, { heads }))
        return fail(tensor_op_errc::input_shape_mismatch);

    if (!shape_is(output, { rows, value_width }))
        return fail(tensor_op_errc::output_shape_mismatch);

    if (aliases(state, { &qkv, &a, &b, &A_log, &dt_bias, &output })
        || aliases(output, { &qkv, &a, &b, &A_log, &dt_bias }))
        return fail(tensor_op_errc::unsupported_aliasing);

    if (!metal_kernels(context).dispatch_gated_delta_rule(
            qkv.buffer(), a.buffer(), b.buffer(), A_log.buffer(), dt_bias.buffer(), state.buffer(),
            output.buffer(), rows, key_heads, heads, key_dim, value_dim, epsilon))
        return fail(tensor_op_errc::backend_failure);

    return { };
}

result<void, tensor_op_errc>
rms_norm_gated(const metal_context& context,
    const metal_tensor& input,
    const metal_tensor& gate,
    const metal_tensor& weight,
    float epsilon,
    metal_tensor& output)
{
    if (input.descriptor().shape().rank() != 2
        || gate.descriptor().shape().rank() != 2
        || weight.descriptor().shape().rank() != 1
        || output.descriptor().shape().rank() != 2)
        return fail(tensor_op_errc::invalid_rank);

    if (!all_f32({ &input, &gate, &weight, &output }))
        return fail(tensor_op_errc::unsupported_dtype);

    if (!std::isfinite(epsilon) || epsilon <= 0)
        return fail(tensor_op_errc::invalid_epsilon);

    const auto dims = input.descriptor().shape().dimensions();
    const auto width = weight.descriptor().shape().dimensions()[0];
    if (dims[1] % width != 0 || !shape_is(gate, { dims[0], dims[1] }))
        return fail(tensor_op_errc::input_shape_mismatch);

    if (!shape_is(output, { dims[0], dims[1] }))
        return fail(tensor_op_errc::output_shape_mismatch);

    if (aliases(output, { &input, &gate, &weight }))
        return fail(tensor_op_errc::unsupported_aliasing);

    if (!metal_kernels(context).dispatch_rms_norm_gated(
            input.buffer(), gate.buffer(), weight.buffer(), output.buffer(),
            input.descriptor().element_count() / width, width, epsilon))
        return fail(tensor_op_errc::backend_failure);

    return { };
}
} // namespace chibillm
