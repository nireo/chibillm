#pragma once

#include <cstdint>
#include <memory>
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

class text_decoder {
public:
    virtual ~text_decoder() = default;
    virtual result<std::string, model_runner_errc> push(token_id token, bool final) = 0;
};

// Adapts one model family to the generic scheduler and serving layers.
class model_runner {
public:
    virtual ~model_runner() = default;

    virtual std::unique_ptr<text_decoder> make_decoder() const;

    virtual result<std::unique_ptr<model_state>, model_runner_errc>
    make_state(scheduler_config config) const;

    [[nodiscard]] virtual const model_info& info() const noexcept = 0;

    [[nodiscard]] virtual result<std::vector<token_id>, model_runner_errc>
    encode_chat(std::span<const chat_message> messages) = 0;

    [[nodiscard]] virtual result<std::string, model_runner_errc>
    decode(std::span<const token_id> tokens) const = 0;

    [[nodiscard]] virtual result<std::vector<token_id>, model_runner_errc>
    execute(const model_batch& batch, model_state& state) = 0;
};

} // namespace chibillm
