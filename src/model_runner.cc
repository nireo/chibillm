#include "model_runner.h"

namespace chibillm {

fake_model_runner::fake_model_runner(token_id output_token) noexcept
    : output_token_(output_token)
{}

result<std::vector<token_id>, model_runner_errc>
fake_model_runner::execute(const model_batch& batch)
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
            || item.logits_index < item.token_offset
            || item.logits_index >= item.token_offset + item.token_count) {
            return fail(model_runner_errc::inconsistent_batch);
        }
    }

    return std::vector<token_id>(batch.items.size(), output_token_);
}

} // namespace chibillm
