#pragma once

#include <cstddef>
#include <cstdint>

#include "result.h"
#include "tensor/tensor_error.h"

namespace chibillm {

enum class dtype : std::uint8_t {
    f32,
    bf16,
    i32,
    u32,
};

[[nodiscard]] inline result<std::size_t, tensor_errc>
element_size(dtype type) noexcept
{
    switch (type) {
    case dtype::f32:
        return sizeof(float);
    case dtype::bf16:
        return sizeof(std::uint16_t);
    case dtype::i32:
        return sizeof(std::int32_t);
    case dtype::u32:
        return sizeof(std::uint32_t);
    }

    return fail(tensor_errc::unsupported_dtype);
}

} // namespace chibillm
