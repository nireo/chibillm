#pragma once

#include <utility>

#include "metal/metal_buffer.h"
#include "metal/metal_context.h"
#include "metal/metal_error.h"
#include "result.h"
#include "tensor/tensor_descriptor.h"

namespace chibillm {

// owns one complete tensor allocation in shared metal storage.
class metal_tensor {
public:
    [[nodiscard]] static result<metal_tensor, metal_error> make(const metal_context& context,
                                                                tensor_descriptor descriptor);

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
