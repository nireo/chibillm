#include "tensor/tensor_ops.h"

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

} // namespace chibillm
