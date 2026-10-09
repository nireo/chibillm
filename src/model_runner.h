#pragma once

#include "error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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

[[nodiscard]] inline std::string_view
error_name(model_runner_errc code) noexcept
{
    static constexpr std::array names {
        "model_runner.empty_batch",     "model_runner.inconsistent_batch",
        "model_runner.invalid_chat",    "model_runner.tokenizer_failure",
        "model_runner.backend_failure",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "model_runner.unknown_error";
}

using model_runner_error = error<model_runner_errc>;

struct chat_message {
    std::string role;
    std::string content;
};

struct encoded_prompt {
    std::vector<token_id> tokens;
    // Stable prefix boundaries, strictly inside the prompt.
    std::vector<std::size_t> checkpoints;
};

struct model_info {
    std::string id;
    std::size_t max_context_tokens;
    token_id eos_token;
};

// Returns the length of the complete UTF-8 prefix, or std::nullopt for invalid bytes.
[[nodiscard]] std::optional<std::size_t> complete_utf8_prefix(std::string_view text);

class text_decoder {
public:
    virtual ~text_decoder() = default;
    virtual result<std::string, model_runner_error> push(token_id token, bool final) = 0;
};

// Adapts one model family to the generic scheduler and serving layers.
class model_runner {
public:
    virtual ~model_runner() = default;

    virtual std::unique_ptr<text_decoder> make_decoder() const;

    virtual result<std::unique_ptr<model_state>, model_runner_error>
    make_state(scheduler_config config) const;

    [[nodiscard]] virtual const model_info& info() const noexcept = 0;

    [[nodiscard]] virtual result<encoded_prompt, model_runner_error>
    encode_chat(std::span<const chat_message> messages) = 0;

    [[nodiscard]] virtual result<std::string, model_runner_error>
    decode(std::span<const token_id> tokens) const = 0;

    [[nodiscard]] virtual result<std::vector<token_id>, model_runner_error>
    execute(const model_batch& batch, model_state& state) = 0;
};

} // namespace chibillm
