#pragma once

#include <cstddef>
#include <limits>
#include <utility>

#include "result.h"
#include "tensor/dtype.h"
#include "tensor/tensor_error.h"
#include "tensor/tensor_shape.h"

namespace chibillm {

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
