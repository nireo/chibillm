#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "result.h"
#include "tensor/tensor_error.h"

namespace chibillm {

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

} // namespace chibillm
