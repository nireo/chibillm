#pragma once

#include "error.h"

#include <cstdint>

namespace chibillm {

enum class tensor_errc : std::uint8_t {
    unsupported_dtype,
    empty_shape,
    zero_dimension,
    element_count_overflow,
    byte_size_overflow,
    axis_out_of_range,
};

[[nodiscard]] inline std::string_view
error_name(tensor_errc code) noexcept
{
    static constexpr std::array names {
        "tensor.unsupported_dtype",      "tensor.empty_shape",        "tensor.zero_dimension",
        "tensor.element_count_overflow", "tensor.byte_size_overflow", "tensor.axis_out_of_range",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "tensor.unknown_error";
}

} // namespace chibillm
