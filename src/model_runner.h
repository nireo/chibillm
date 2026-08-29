#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "model_batch.h"
#include "result.h"

namespace chibillm {

enum class model_runner_errc : std::uint8_t {
    empty_batch,
    inconsistent_batch,
    invalid_chat,
    tokenizer_failure,
    backend_failure,
};

struct chat_message {
    std::string role;
    std::string content;
};

struct model_info {
    std::string id;
    std::size_t max_context_tokens;
    token_id eos_token;
};

// Adapts one model family to the generic scheduler and serving layers.
class model_runner {
public:
    virtual ~model_runner() = default;

    [[nodiscard]] virtual const model_info& info() const noexcept = 0;

    [[nodiscard]] virtual result<std::vector<token_id>, model_runner_errc>
    encode_chat(std::span<const chat_message> messages) = 0;

    [[nodiscard]] virtual result<std::string, model_runner_errc>
    decode(std::span<const token_id> tokens) const = 0;

    [[nodiscard]] virtual result<std::vector<token_id>, model_runner_errc>
    execute(const model_batch& batch) = 0;
};

// returns a fixed token while the real model backend is unavailable.
class fake_model_runner final : public model_runner {
public:
    explicit fake_model_runner(token_id output_token) noexcept;

    [[nodiscard]] const model_info& info() const noexcept override;

    [[nodiscard]] result<std::vector<token_id>, model_runner_errc>
    encode_chat(std::span<const chat_message> messages) override;

    [[nodiscard]] result<std::string, model_runner_errc>
    decode(std::span<const token_id> tokens) const override;

    [[nodiscard]] result<std::vector<token_id>, model_runner_errc>
    execute(const model_batch& batch) override;

private:
    model_info info_ { .id = "fake", .max_context_tokens = 4096, .eos_token = -1 };
    token_id output_token_;
};

} // namespace chibillm
