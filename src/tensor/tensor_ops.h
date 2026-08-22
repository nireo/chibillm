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
    backend_failure,
};

// computes lhs [m, k] times rhs [k, n] into output [m, n].
[[nodiscard]] result<void, tensor_op_errc> matmul(const metal_context& context,
                                                  const metal_tensor& lhs,
                                                  const metal_tensor& rhs,
                                                  metal_tensor& output);

} // namespace chibillm
