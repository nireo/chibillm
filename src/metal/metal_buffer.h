#pragma once

#include <cstddef>
#include <memory>
#include <span>

namespace chibillm {

class metal_context;

// owns one cpu-visible metal buffer.
class metal_buffer {
public:
    metal_buffer(const metal_buffer&) = delete;
    metal_buffer& operator=(const metal_buffer&) = delete;
    metal_buffer(metal_buffer&&) noexcept;
    metal_buffer& operator=(metal_buffer&&) noexcept;
    ~metal_buffer();

    [[nodiscard]] std::size_t size_bytes() const noexcept;
    [[nodiscard]] std::span<std::byte> bytes() noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

private:
    friend class metal_context;

    struct implementation;

    explicit metal_buffer(std::unique_ptr<implementation> implementation) noexcept;

    std::unique_ptr<implementation> implementation_;
};

} // namespace chibillm
