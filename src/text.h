#pragma once
#include <optional>
#include <string_view>

namespace chibillm {
std::optional<std::size_t> complete_utf8_prefix(std::string_view text);
}
