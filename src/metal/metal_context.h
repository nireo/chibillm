#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string_view>

#include "metal/metal_buffer.h"
#include "metal/metal_error.h"
#include "result.h"

namespace chibillm {

class metal_kernels;

// owns the metal device, queue, shader library, and compute pipelines.
class metal_context {
public:
    [[nodiscard]] static result<metal_context, metal_error> make(std::string_view shader_source);

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
