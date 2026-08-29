#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "model_runner.h"
#include "result.h"

namespace chibillm {

struct server_config {
    std::string host { "127.0.0.1" };
    std::uint16_t port { 8000 };
    std::size_t max_sequences { 4 };
    std::size_t max_pending_requests { 64 };
    std::size_t max_batch_tokens { 128 };
    std::size_t kv_block_count { 64 };
    std::size_t kv_block_size { 16 };
    std::size_t default_max_completion_tokens { 256 };
};

enum class server_errc : std::uint8_t {
    invalid_config,
    engine_creation_failed,
    bind_failed,
    listen_failed,
};

class openai_server {
public:
    [[nodiscard]] static result<std::unique_ptr<openai_server>, server_errc>
    make(model_runner& runner, server_config config = {});

    ~openai_server();

    openai_server(const openai_server&) = delete;
    openai_server& operator=(const openai_server&) = delete;

    [[nodiscard]] std::uint16_t port() const noexcept;
    [[nodiscard]] result<void, server_errc> run();
    void stop() noexcept;

private:
    struct implementation;

    explicit openai_server(std::unique_ptr<implementation> implementation) noexcept;

    std::unique_ptr<implementation> implementation_;
};

} // namespace chibillm
