#pragma once
#include "model_runner.h"

namespace chibillm {
class fake_model_runner final : public model_runner {
public:
    explicit fake_model_runner(token_id output_token) noexcept;

    [[nodiscard]] const model_info& info() const noexcept override;

    [[nodiscard]] result<std::vector<token_id>, model_runner_errc>
    encode_chat(std::span<const chat_message> messages) override;

    [[nodiscard]] result<std::string, model_runner_errc>
    decode(std::span<const token_id> tokens) const override;

    [[nodiscard]] result<std::vector<token_id>, model_runner_errc>
    execute(const model_batch& batch, model_state& state) override;

private:
    model_info info_ { .id = "fake", .max_context_tokens = 4096, .eos_token = -1 };
    token_id output_token_;
};

} // namespace chibillm
