#include "tensor/tensor_ops.h"
#pragma once

#include <cstdint>
#include <span>

#include "metal/metal_context.h"
#include "metal/metal_tensor.h"
#include "result.h"
#include "seq.h"

namespace chibillm {

[[nodiscard]] result<metal_tensor, tensor_op_errc> embed_tokens(const metal_context& context,
                                                                const metal_tensor& weight,
                                                                std::span<const token_id> tokens);

} // namespace chibillm
