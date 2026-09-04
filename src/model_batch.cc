#include "model_batch.h"

#include <limits>
#include <unordered_set>

namespace chibillm {

bool
model_batch::empty() const noexcept
{
    return items.empty();
}

std::size_t
model_batch::sequence_count() const noexcept
{
    return items.size();
}

std::size_t
model_batch::token_count() const noexcept
{
    return tokens.size();
}

result<model_batch, model_batch_errc>
build_model_batch(const scheduled_batch& scheduled, const scheduler& engine)
{
    if (scheduled.empty()) {
        return fail(model_batch_errc::empty_batch);
    }

    if (scheduled.phase != batch_phase::prefill && scheduled.phase != batch_phase::decode) {
        return fail(model_batch_errc::invalid_sequence_state);
    }

    std::unordered_set<seq_id> sequence_ids;
    sequence_ids.reserve(scheduled.items.size());

    const auto block_size = engine.config().kv_block_size;
    const auto max_position = std::numeric_limits<position_id>::max();

    // validate the complete reservation before constructing partial output.
    for (const auto& item : scheduled.items) {
        if (item.token_count == 0) {
            return fail(model_batch_errc::zero_token_count);
        }

        if (!sequence_ids.insert(item.id).second) {
            return fail(model_batch_errc::duplicate_sequence_id);
        }

        const auto* sequence = engine.find_sequence(item.id);
        if (sequence == nullptr) {
            return fail(model_batch_errc::unknown_sequence);
        }

        const auto expected_status =
            scheduled.phase == batch_phase::prefill ? seq_status::waiting : seq_status::running;
        if (sequence->status() != expected_status) {
            return fail(model_batch_errc::invalid_sequence_state);
        }

        if (item.token_count != sequence->scheduled_token_count()) {
            return fail(model_batch_errc::scheduled_token_count_mismatch);
        }

        const auto token_count = sequence->token_count();
        const auto token_offset = sequence->cached_token_count();
        if (token_offset > token_count || item.token_count > token_count - token_offset) {
            return fail(model_batch_errc::token_range_out_of_bounds);
        }

        if (sequence->block_size() != block_size || engine.cache().block_size() != block_size) {
            return fail(model_batch_errc::incompatible_block_size);
        }

        if (sequence->block_table().size() < sequence->logical_block_count()) {
            return fail(model_batch_errc::incomplete_block_table);
        }

        if (token_offset > max_position || item.token_count - 1 > max_position - token_offset) {
            return fail(model_batch_errc::position_out_of_range);
        }
    }

    model_batch output {
        .id = scheduled.id,
        .phase = scheduled.phase,
        .kv_block_size = block_size,
        .tokens = {},
        .positions = {},
        .items = {},
    };

    const auto total_token_count = scheduled.token_count();
    output.tokens.reserve(total_token_count);
    output.positions.reserve(total_token_count);
    output.items.reserve(scheduled.items.size());

    for (const auto& item : scheduled.items) {
        const auto& sequence = *engine.find_sequence(item.id);
        const auto token_offset = output.tokens.size();
        const auto sequence_offset = sequence.cached_token_count();
        const auto sequence_tokens = sequence.tokens().subspan(sequence_offset, item.token_count);

        output.tokens.insert(output.tokens.end(), sequence_tokens.begin(), sequence_tokens.end());

        for (std::size_t index = 0; index < item.token_count; ++index) {
            output.positions.push_back(static_cast<position_id>(sequence_offset + index));
        }

        const auto block_table = sequence.block_table();
        output.items.push_back(model_batch_item {
            .id = item.id,
            .token_offset = token_offset,
            .token_count = item.token_count,
            .logits_index = token_offset + item.token_count - 1,
            .block_table = std::vector<block_id>(block_table.begin(), block_table.end()),
        });
    }

    return output;
}

} // namespace chibillm
