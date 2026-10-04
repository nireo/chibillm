#pragma once

#include "error.h"

#include <cstdint>
#include <string>

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

} // namespace chibillm
