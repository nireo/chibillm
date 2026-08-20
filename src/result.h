#pragma once

#include <expected>
#include <utility>

namespace chibillm {

template <typename T, typename E> using result = std::expected<T, E>;

template <typename E>
[[nodiscard]] constexpr std::unexpected<E>
fail(E error)
{
    return std::unexpected<E> { std::move(error) };
}

} // namespace chibillm
