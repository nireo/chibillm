#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "error.h"
#include "result.h"

namespace chibillm {

enum class metal_errc : std::uint8_t {
    no_device,
    command_queue_creation_failed,
    kernel_library_creation_failed,
    kernel_function_not_found,
    pipeline_creation_failed,
    buffer_creation_failed,
    command_buffer_creation_failed,
    command_encoder_creation_failed,
    invalid_input,
    execution_failed,
};

[[nodiscard]] inline std::string_view
error_name(metal_errc code) noexcept
{
    static constexpr std::array names {
        "metal.no_device",
        "metal.command_queue_creation_failed",
        "metal.kernel_library_creation_failed",
        "metal.kernel_function_not_found",
        "metal.pipeline_creation_failed",
        "metal.buffer_creation_failed",
        "metal.command_buffer_creation_failed",
        "metal.command_encoder_creation_failed",
        "metal.invalid_input",
        "metal.execution_failed",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "metal.unknown_error";
}

struct metal_error {
    metal_errc code;
    std::string message;
};

[[nodiscard]] inline std::string
describe_error(const metal_error& value)
{
    return describe_error(value.code) + ": " + value.message;
}

class metal_context;
class metal_kernels;

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
    friend class metal_kernels;

    struct implementation;

    explicit metal_buffer(std::unique_ptr<implementation> implementation) noexcept;

    std::unique_ptr<implementation> implementation_;
};

// owns the metal device, queue, kernel library, and compute pipelines.
class metal_context {
public:
    [[nodiscard]] static result<metal_context, metal_error> make(std::string_view kernel_source);

    metal_context(const metal_context&) = delete;
    metal_context& operator=(const metal_context&) = delete;
    metal_context(metal_context&&) noexcept;
    metal_context& operator=(metal_context&&) noexcept;
    ~metal_context();

    [[nodiscard]] std::string_view device_name() const noexcept;

    // opens one command buffer whose compute encoder collects every dispatch until
    // end_compute_pass(). Dispatches made while a pass is open are appended to it;
    // they no longer wait on the GPU individually.
    [[nodiscard]] result<void, metal_error> begin_compute_pass();

    // commits the open command buffer and waits once for all of its kernels.
    [[nodiscard]] result<void, metal_error> end_compute_pass();

    // commits (and waits on) whatever kernels were already encoded into the open
    // pass, then closes it. Used when a batched operation fails midway so GPU and
    // shared-memory state still settle before returning to the caller.
    void abort_compute_pass() noexcept;

    [[nodiscard]] result<metal_buffer, metal_error>
    make_shared_buffer(std::size_t size_bytes) const;

private:
    friend class metal_kernels;
    struct implementation;
    explicit metal_context(std::unique_ptr<implementation> implementation) noexcept;
    std::unique_ptr<implementation> implementation_;
};

class compute_pass {
public:
    explicit compute_pass(metal_context& context)
        : context_(context)
    {}

    ~compute_pass()
    {
        if (open_)
            context_.abort_compute_pass();
    }

    compute_pass(const compute_pass&) = delete;
    compute_pass& operator=(const compute_pass&) = delete;

    result<void, metal_error>
    begin()
    {
        auto result = context_.begin_compute_pass();
        open_ = result.has_value();
        return result;
    }

    result<void, metal_error>
    finish()
    {
        auto result = context_.end_compute_pass();
        open_ = false;
        return result;
    }

private:
    metal_context& context_;
    bool open_ = false;
};

} // namespace chibillm
