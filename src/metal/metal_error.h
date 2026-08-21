#pragma once

#include <cstdint>
#include <string>

namespace chibillm {

enum class metal_errc : std::uint8_t {
    no_device,
    command_queue_creation_failed,
    shader_library_creation_failed,
    shader_function_not_found,
    pipeline_creation_failed,
    buffer_creation_failed,
    command_buffer_creation_failed,
    command_encoder_creation_failed,
    invalid_input,
    execution_failed,
};

struct metal_error {
    metal_errc code;
    std::string message;
};

} // namespace chibillm
