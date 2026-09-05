#include "fake_model_runner.h"
#include <limits>

namespace chibillm {
fake_model_runner::fake_model_runner(token_id output_token) noexcept
    : output_token_(output_token)
{}

const model_info&
fake_model_runner::info() const noexcept
{
    return info_;
}

result<std::vector<token_id>, model_runner_errc>
fake_model_runner::encode_chat(std::span<const chat_message> messages)
{
    if (messages.empty()) {
        return fail(model_runner_errc::invalid_chat);
    }

    std::vector<token_id> tokens;
    for (const auto& message : messages) {
        for (const auto byte : message.content) {
            tokens.push_back(static_cast<unsigned char>(byte));
        }
    }
    if (tokens.empty()) {
        tokens.push_back(0);
    }
    return tokens;
}

result<std::string, model_runner_errc>
fake_model_runner::decode(std::span<const token_id> tokens) const
{
    std::string text;
    text.reserve(tokens.size());
    for (const auto token : tokens) {
        if (token < 0 || token > std::numeric_limits<unsigned char>::max()) {
            return fail(model_runner_errc::tokenizer_failure);
        }
        text.push_back(static_cast<char>(token));
    }
    return text;
}

result<std::vector<token_id>, model_runner_errc>
fake_model_runner::execute(const model_batch& batch, model_state&)
{
    if (batch.empty()) {
        return fail(model_runner_errc::empty_batch);
    }

    if (batch.tokens.size() != batch.positions.size()) {
        return fail(model_runner_errc::inconsistent_batch);
    }

    for (const auto& item : batch.items) {
        if (item.token_count == 0
            || item.token_offset > batch.tokens.size()
            || item.token_count > batch.tokens.size() - item.token_offset
            || (item.logits_index
                && (*item.logits_index < item.token_offset
                    || *item.logits_index >= item.token_offset + item.token_count))) {
            return fail(model_runner_errc::inconsistent_batch);
        }
    }

    return std::vector<token_id>(batch.sample_count(), output_token_);
}

} // namespace chibillm
