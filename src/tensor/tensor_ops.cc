#include "tensor/tensor_ops.h"
#include "tensor/dtype.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace chibillm {

result<void, tensor_op_errc>
matmul(const metal_context& context,
       const metal_tensor& lhs,
       const metal_tensor& rhs,
       metal_tensor& output)
{
    const auto& lshape = lhs.descriptor().shape();
    const auto& rshape = rhs.descriptor().shape();
    const auto& oshape = output.descriptor().shape();

    if (lshape.rank() != 2 || rshape.rank() != 2 || oshape.rank() != 2) {
        return fail(tensor_op_errc::invalid_rank);
    }

    if (lhs.descriptor().type() != dtype::f32
        || rhs.descriptor().type() != dtype::f32
        || output.descriptor().type() != dtype::f32) {
        return fail(tensor_op_errc::unsupported_dtype);
    }

    const auto m = lshape.dimensions()[0];
    const auto k = lshape.dimensions()[1];
    const auto n = rshape.dimensions()[1];

    if (k != rshape.dimensions()[0]) {
        return fail(tensor_op_errc::inner_dimension_mismatch);
    }

    if (oshape.dimensions()[0] != m || oshape.dimensions()[1] != n) {
        return fail(tensor_op_errc::output_shape_mismatch);
    }

    const auto dispatched =
        context.dispatch_matmul(lhs.buffer(), rhs.buffer(), output.buffer(), m, k, n);
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure);
    }

    return {};
}

result<void, tensor_op_errc>
linear(const metal_context& context,
       const metal_tensor& input,
       const metal_tensor& weight,
       metal_tensor& output)
{
    const auto& input_shape = input.descriptor().shape();
    const auto& weight_shape = weight.descriptor().shape();
    const auto& output_shape = output.descriptor().shape();

    if (input_shape.rank() != 2 || weight_shape.rank() != 2 || output_shape.rank() != 2) {
        return fail(tensor_op_errc::invalid_rank);
    }

    if (input.descriptor().type() != dtype::f32
        || weight.descriptor().type() != dtype::bf16
        || output.descriptor().type() != dtype::f32) {
        return fail(tensor_op_errc::unsupported_dtype);
    }

    const auto rows = input_shape.dimensions()[0];
    const auto input_features = input_shape.dimensions()[1];
    const auto output_features = weight_shape.dimensions()[0];

    if (weight_shape.dimensions()[1] != input_features) {
        return fail(tensor_op_errc::inner_dimension_mismatch);
    }

    if (output_shape.dimensions()[0] != rows || output_shape.dimensions()[1] != output_features) {
        return fail(tensor_op_errc::output_shape_mismatch);
    }

    const auto dispatched = context.dispatch_linear_bf16(
        input.buffer(), weight.buffer(), output.buffer(), rows, input_features, output_features);
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure);
    }

    return {};
}

result<void, tensor_op_errc>
embedding_lookup(const metal_context& context,
                 const metal_tensor& token_ids,
                 const metal_tensor& weight,
                 metal_tensor& output)
{
    const auto& token_shape = token_ids.descriptor().shape();
    const auto& weight_shape = weight.descriptor().shape();
    const auto& output_shape = output.descriptor().shape();

    if (token_shape.rank() != 1 || weight_shape.rank() != 2 || output_shape.rank() != 2) {
        return fail(tensor_op_errc::invalid_rank);
    }

    if (token_ids.descriptor().type() != dtype::i32
        || weight.descriptor().type() != dtype::bf16
        || output.descriptor().type() != dtype::f32) {
        return fail(tensor_op_errc::unsupported_dtype);
    }

    const auto token_count = token_shape.dimensions()[0];
    const auto vocabulary_size = weight_shape.dimensions()[0];
    const auto hidden_size = weight_shape.dimensions()[1];

    if (output_shape.dimensions()[0] != token_count
        || output_shape.dimensions()[1] != hidden_size) {
        return fail(tensor_op_errc::output_shape_mismatch);
    }

    const auto token_bytes = token_ids.buffer().bytes();
    for (std::size_t index = 0; index < token_count; ++index) {
        std::int32_t token = 0;
        std::memcpy(&token, token_bytes.data() + index * sizeof(token), sizeof(token));

        if (token < 0 || static_cast<std::size_t>(token) >= vocabulary_size) {
            return fail(tensor_op_errc::token_out_of_range);
        }
    }

    const auto dispatched = context.dispatch_embedding_bf16(
        token_ids.buffer(), weight.buffer(), output.buffer(), token_count, hidden_size);
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure);
    }

    return {};
}

result<void, tensor_op_errc>
rms_norm(const metal_context& context,
         const metal_tensor& input,
         const metal_tensor& weight,
         float epsilon,
         metal_tensor& output)
{
    const auto& input_shape = input.descriptor().shape();
    const auto& weight_shape = weight.descriptor().shape();
    const auto& output_shape = output.descriptor().shape();

    if (input_shape.rank() != 2 || weight_shape.rank() != 1 || output_shape.rank() != 2) {
        return fail(tensor_op_errc::invalid_rank);
    }

    if (input.descriptor().type() != dtype::f32
        || weight.descriptor().type() != dtype::bf16
        || output.descriptor().type() != dtype::f32) {
        return fail(tensor_op_errc::unsupported_dtype);
    }

    const auto rows = input_shape.dimensions()[0];
    const auto hidden_size = input_shape.dimensions()[1];

    if (weight_shape.dimensions()[0] != hidden_size) {
        return fail(tensor_op_errc::inner_dimension_mismatch);
    }

    if (output_shape.dimensions()[0] != rows || output_shape.dimensions()[1] != hidden_size) {
        return fail(tensor_op_errc::output_shape_mismatch);
    }

    if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
        return fail(tensor_op_errc::invalid_epsilon);
    }

    const auto dispatched = context.dispatch_rms_norm_bf16(
        input.buffer(), weight.buffer(), output.buffer(), rows, hidden_size, epsilon);
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure);
    }

    return {};
}

result<void, tensor_op_errc>
silu_mul(const metal_context& context,
         const metal_tensor& gate,
         const metal_tensor& up,
         metal_tensor& output)
{
    const auto& gate_descriptor = gate.descriptor();
    const auto& up_descriptor = up.descriptor();
    const auto& output_descriptor = output.descriptor();
    const auto& gate_shape = gate_descriptor.shape();
    const auto& up_shape = up_descriptor.shape();
    const auto& output_shape = output_descriptor.shape();

    if (gate_shape.rank() != 2 || up_shape.rank() != 2 || output_shape.rank() != 2) {
        return fail(tensor_op_errc::invalid_rank);
    }

    if (gate_descriptor.type() != dtype::f32
        || up_descriptor.type() != dtype::f32
        || output_descriptor.type() != dtype::f32) {
        return fail(tensor_op_errc::unsupported_dtype);
    }

    if (gate_shape.dimensions()[0] != up_shape.dimensions()[0]
        || gate_shape.dimensions()[1] != up_shape.dimensions()[1]) {
        return fail(tensor_op_errc::input_shape_mismatch);
    }

    if (output_shape.dimensions()[0] != gate_shape.dimensions()[0]
        || output_shape.dimensions()[1] != gate_shape.dimensions()[1]) {
        return fail(tensor_op_errc::output_shape_mismatch);
    }

    const auto dispatched = context.dispatch_silu_mul_f32(
        gate.buffer(), up.buffer(), output.buffer(), gate_shape.element_count());
    if (!dispatched) {
        return fail(tensor_op_errc::backend_failure);
    }

    return {};
}

} // namespace chibillm
