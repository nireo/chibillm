#pragma once

#include <array>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace chibillm {

// Keep the code available for control flow; detail carries context from lower layers.
// Empty detail requires no allocation, including for allocation-failure errors.
template <typename Code> struct error {
    Code code;
    std::string detail;

    error(Code code) noexcept
        : code(code)
    {}

    error(Code code, std::string detail)
        : code(code)
        , detail(std::move(detail))
    {}

    [[nodiscard]] bool
    operator==(Code other) const noexcept
    {
        return code == other;
    }
};

template <typename Code>
    requires std::is_enum_v<Code>
[[nodiscard]] std::string
describe_error(Code code)
{
    return std::string(error_name(code));
}

template <typename Code>
[[nodiscard]] std::string
describe_error(const error<Code>& value)
{
    auto description = describe_error(value.code);
    if (!value.detail.empty()) {
        description += ": ";
        description += value.detail;
    }
    return description;
}

[[nodiscard]] inline std::string
describe_error(std::string_view message)
{
    return std::string(message);
}

template <typename Code>
[[nodiscard]] error<Code>
at_stage(error<Code> value, std::string_view stage)
{
    value.detail =
        value.detail.empty() ? std::string(stage) : std::string(stage) + ": " + value.detail;
    return value;
}

} // namespace chibillm
