#pragma once

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

} // namespace chibillm
