#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "metal/metal_buffer.h"
#include "metal/metal_context.h"
#include "metal/metal_error.h"
#include "result.h"
#include "tensor/tensor_descriptor.h"

namespace chibillm {

enum class metal_tensor_errc : std::uint8_t {
    invalid_descriptor,
    allocation_failed,
};

// owns one complete tensor allocation in shared metal storage.
class metal_tensor {
public:
    [[nodiscard]] static result<metal_tensor, metal_error> make(const metal_context& context,
                                                                tensor_descriptor descriptor);

    [[nodiscard]] static result<metal_tensor, metal_tensor_errc>
    make(const metal_context& context, dtype type, std::vector<std::size_t> dimensions);

    metal_tensor(const metal_tensor&) = delete;
    metal_tensor& operator=(const metal_tensor&) = delete;
    metal_tensor(metal_tensor&&) noexcept = default;
    metal_tensor& operator=(metal_tensor&&) noexcept = default;

    [[nodiscard]] const tensor_descriptor& descriptor() const noexcept;
    [[nodiscard]] metal_buffer& buffer() noexcept;
    [[nodiscard]] const metal_buffer& buffer() const noexcept;

private:
    metal_tensor(tensor_descriptor descriptor, metal_buffer buffer);

    tensor_descriptor descriptor_;
    metal_buffer buffer_;
};

inline metal_tensor::metal_tensor(tensor_descriptor descriptor, metal_buffer buffer)
    : descriptor_(std::move(descriptor))
    , buffer_(std::move(buffer))
{}

inline result<metal_tensor, metal_error>
metal_tensor::make(const metal_context& context, tensor_descriptor descriptor)
{
    auto buffer = context.make_shared_buffer(descriptor.size_bytes());
    if (!buffer) {
        return fail(std::move(buffer.error()));
    }

    return metal_tensor {
        std::move(descriptor),
        std::move(*buffer),
    };
}

inline result<metal_tensor, metal_tensor_errc>
metal_tensor::make(const metal_context& context, dtype type, std::vector<std::size_t> dimensions)
{
    auto shape = tensor_shape::make(std::move(dimensions));
    if (!shape) {
        return fail(metal_tensor_errc::invalid_descriptor);
    }
    auto descriptor = tensor_descriptor::make(type, std::move(*shape));
    if (!descriptor) {
        return fail(metal_tensor_errc::invalid_descriptor);
    }
    auto tensor = make(context, std::move(*descriptor));
    if (!tensor) {
        return fail(metal_tensor_errc::allocation_failed);
    }
    return std::move(*tensor);
}

inline const tensor_descriptor&
metal_tensor::descriptor() const noexcept
{
    return descriptor_;
}

inline metal_buffer&
metal_tensor::buffer() noexcept
{
    return buffer_;
}

inline const metal_buffer&
metal_tensor::buffer() const noexcept
{
    return buffer_;
}

} // namespace chibillm
