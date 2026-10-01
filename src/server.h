#pragma once

#include "error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "result.h"
#include "serving_runtime.h"

namespace chibillm {

struct server_config {
    std::string host { "127.0.0.1" };
    std::uint16_t port { 8000 };
    serving_config runtime;
    std::size_t default_max_completion_tokens { 256 };
};

enum class server_errc : std::uint8_t {
    invalid_config,
    engine_creation_failed,
    bind_failed,
    listen_failed,
};

[[nodiscard]] inline std::string_view
error_name(server_errc code) noexcept
{
    static constexpr std::array names {
        "server.invalid_config",
        "server.engine_creation_failed",
        "server.bind_failed",
        "server.listen_failed",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "server.unknown_error";
}

using server_error = error<server_errc>;

class openai_server {
public:
    [[nodiscard]] static result<std::unique_ptr<openai_server>, server_error>
    make(model_runner& runner, server_config config = {});

    ~openai_server();

    openai_server(const openai_server&) = delete;
    openai_server& operator=(const openai_server&) = delete;

    [[nodiscard]] std::uint16_t port() const noexcept;
    [[nodiscard]] result<void, server_error> run();
    void stop() noexcept;

private:
    struct implementation;

    explicit openai_server(std::unique_ptr<implementation> implementation) noexcept;

    std::unique_ptr<implementation> implementation_;
};

} // namespace chibillm
