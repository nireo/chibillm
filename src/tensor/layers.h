#pragma once
#include "tensor/tensor_ops.h"

namespace chibillm {
result<metal_tensor, tensor_op_errc> normalized_swiglu(const metal_context& context,
                                                       const metal_tensor& norm,
                                                       const metal_tensor& gateup,
                                                       const metal_tensor& down,
                                                       float epsilon,
                                                       const metal_tensor& hidden_states);
}
