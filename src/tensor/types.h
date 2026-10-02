#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "error.h"
#include "result.h"

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

class bf16 {
public:
    [[nodiscard]] static bf16 from_bits(std::uint16_t bits) noexcept;
    [[nodiscard]] static bf16 from_float(float value) noexcept;

    [[nodiscard]] std::uint16_t bits() const noexcept;
    [[nodiscard]] float to_float() const noexcept;

private:
    explicit bf16(std::uint16_t bits) noexcept;

    std::uint16_t bits_;
};

inline bf16::bf16(std::uint16_t bits) noexcept
    : bits_(bits)
{}

inline bf16
bf16::from_bits(std::uint16_t bits) noexcept
{
    return bf16 { bits };
}

inline bf16
bf16::from_float(float value) noexcept
{
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);

    // NaN: preserve the sign/payload bits that fit and ensure the bf16 mantissa remains non-zero.
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return bf16(static_cast<std::uint16_t>((bits >> 16) | 0x0040u));
    }

    // round to nearest, ties to even.
    const std::uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;

    return bf16(static_cast<std::uint16_t>(bits >> 16));
}

inline std::uint16_t
bf16::bits() const noexcept
{
    return bits_;
}

inline float
bf16::to_float() const noexcept
{
    const auto float_bits = static_cast<std::uint32_t>(bits_) << 16U;
    return std::bit_cast<float>(float_bits);
}

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

class tensor_shape {
public:
    [[nodiscard]] static result<tensor_shape, tensor_errc>
    make(std::vector<std::size_t> dimensions);

    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] std::span<const std::size_t> dimensions() const noexcept;
    [[nodiscard]] std::size_t element_count() const noexcept;
    [[nodiscard]] result<std::size_t, tensor_errc> dimension(std::size_t axis) const noexcept;

private:
    tensor_shape(std::vector<std::size_t> dimensions, std::size_t element_count);

    std::vector<std::size_t> dimensions_;
    std::size_t element_count_;
};

inline tensor_shape::tensor_shape(std::vector<std::size_t> dimensions, std::size_t element_count)
    : dimensions_(std::move(dimensions))
    , element_count_(element_count)
{}

inline result<tensor_shape, tensor_errc>
tensor_shape::make(std::vector<std::size_t> dimensions)
{
    if (dimensions.size() == 0) {
        return fail(tensor_errc::empty_shape);
    }

    std::size_t count = 1;
    for (const auto dim : dimensions) {
        if (dim == 0) {
            return fail(tensor_errc::zero_dimension);
        }

        if (count > std::numeric_limits<std::size_t>::max() / dim) {
            return fail(tensor_errc::element_count_overflow);
        }

        count *= dim;
    }

    return tensor_shape(std::move(dimensions), count);
}

inline std::size_t
tensor_shape::rank() const noexcept
{
    return dimensions_.size();
}

inline std::span<const std::size_t>
tensor_shape::dimensions() const noexcept
{
    return dimensions_;
}

inline std::size_t
tensor_shape::element_count() const noexcept
{
    return element_count_;
}

inline result<std::size_t, tensor_errc>
tensor_shape::dimension(std::size_t axis) const noexcept
{
    if (axis >= dimensions_.size()) {
        return fail(tensor_errc::axis_out_of_range);
    }

    return dimensions_[axis];
}

class tensor_descriptor {
public:
    [[nodiscard]] static result<tensor_descriptor, tensor_errc> make(dtype type,
                                                                     tensor_shape shape);

    [[nodiscard]] dtype type() const noexcept;
    [[nodiscard]] const tensor_shape& shape() const noexcept;
    [[nodiscard]] std::size_t element_count() const noexcept;
    [[nodiscard]] std::size_t size_bytes() const noexcept;

private:
    tensor_descriptor(dtype type, tensor_shape shape, std::size_t size_bytes);

    dtype type_;
    tensor_shape shape_;
    std::size_t size_bytes_;
};

inline tensor_descriptor::tensor_descriptor(dtype type, tensor_shape shape, std::size_t size_bytes)
    : type_(type)
    , shape_(std::move(shape))
    , size_bytes_(size_bytes)
{}

inline result<tensor_descriptor, tensor_errc>
tensor_descriptor::make(dtype type, tensor_shape shape)
{
    const auto elem_size = element_size(type);
    if (!elem_size) {
        return fail(elem_size.error());
    }

    // this is maybe stupid lol
    if (*elem_size > std::numeric_limits<std::size_t>::max() / shape.element_count()) {
        return fail(tensor_errc::byte_size_overflow);
    }
    const std::size_t size = *elem_size * shape.element_count();

    return tensor_descriptor(type, std::move(shape), size);
}

inline dtype
tensor_descriptor::type() const noexcept
{
    return type_;
}

inline const tensor_shape&
tensor_descriptor::shape() const noexcept
{
    return shape_;
}

inline std::size_t
tensor_descriptor::element_count() const noexcept
{
    return shape_.element_count();
}

inline std::size_t
tensor_descriptor::size_bytes() const noexcept
{
    return size_bytes_;
}

} // namespace chibillm
