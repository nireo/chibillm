#pragma once

#include <expected>
#include <utility>

#include "error.h"

namespace chibillm {

template <typename T, typename E> using result = std::expected<T, E>;

template <typename E>
[[nodiscard]] constexpr std::unexpected<E>
fail(E error)
{
    return std::unexpected<E> { std::move(error) };
}

// Translate a code at a boundary without discarding the cause or failing stage.
template <typename Code, typename Cause>
[[nodiscard]] std::unexpected<error<Code>>
fail(Code code, const Cause& cause, std::string_view stage = {})
{
    auto detail = describe_error(cause);
    if (!stage.empty())
        detail = std::string(stage) + ": " + detail;
    return std::unexpected<error<Code>> { error<Code> { code, std::move(detail) } };
}

} // namespace chibillm

// Propagates the error of an expected-like expression from the enclosing function.
#define CL_TRY(expression)                                                                         \
    do {                                                                                           \
        auto cl_try_result = (expression);                                                         \
        if (!cl_try_result)                                                                        \
            return ::chibillm::fail(cl_try_result.error());                                        \
    } while (false)

// Assigns the value of an expected-like expression or returns its error from
// the enclosing function. The scoped temporary makes repeated uses safe.
#define CL_TRY_ASSIGN(destination, expression)                                                     \
    do {                                                                                           \
        auto cl_try_result = (expression);                                                         \
        if (!cl_try_result)                                                                        \
            return ::chibillm::fail(cl_try_result.error());                                        \
        destination = std::move(*cl_try_result);                                                   \
    } while (false)
