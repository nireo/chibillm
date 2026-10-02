#pragma once

#include <initializer_list>

#include "tensor/tensor_ops.h"

namespace chibillm {

struct tensor_layout {
    const metal_tensor& tensor;
    std::size_t rank;
    dtype type;
};

// Check every rank before inspecting dimensions or reporting dtype errors.
[[nodiscard]] inline result<void, tensor_op_error>
validate_tensor_layouts(std::initializer_list<tensor_layout> layouts)
{
    for (const auto& layout : layouts) {
        if (layout.tensor.descriptor().shape().rank() != layout.rank)
            return fail(tensor_op_errc::invalid_rank);
    }
    for (const auto& layout : layouts) {
        if (layout.tensor.descriptor().type() != layout.type)
            return fail(tensor_op_errc::unsupported_dtype);
    }
    return {};
}

} // namespace chibillm
