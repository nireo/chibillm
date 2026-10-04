#pragma once

#include <cstdint>

namespace chibillm {

enum class weight_quantization : std::uint8_t {
    none,
    q4
};

} // namespace chibillm
