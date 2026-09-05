#include "model_batch.h"

#include <algorithm>
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

std::size_t
model_batch::sample_count() const noexcept
{
    return std::count_if(items.begin(), items.end(),
                         [](const auto& item) { return item.logits_index.has_value(); });
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

    const auto block_size = engine.state().block_size();
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
        const auto token_offset = sequence->processed_token_count();
        if (token_offset > token_count || item.token_count > token_count - token_offset) {
            return fail(model_batch_errc::token_range_out_of_bounds);
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
        const auto sequence_offset = sequence.processed_token_count();
        const auto sequence_tokens = sequence.tokens().subspan(sequence_offset, item.token_count);

        output.tokens.insert(output.tokens.end(), sequence_tokens.begin(), sequence_tokens.end());

        for (std::size_t index = 0; index < item.token_count; ++index) {
            output.positions.push_back(static_cast<position_id>(sequence_offset + index));
        }

        const auto block_table = engine.state().resources(item.id).blocks;
        output.items.push_back(model_batch_item {
            .id = item.id,
            .token_offset = token_offset,
            .token_count = item.token_count,
            .logits_index =
                item.sample ? std::optional(token_offset + item.token_count - 1) : std::nullopt,
            .block_table = std::vector<block_id>(block_table.begin(), block_table.end()),
        });
    }

    return output;
}

result<paged_batch_metadata, model_batch_errc>
prepare_paged_batch(const model_batch& batch,
                    std::size_t max_positions,
                    std::size_t block_count,
                    std::size_t block_size)
{
    if (batch.empty()) {
        return fail(model_batch_errc::empty_batch);
    }
    if (!block_size || !block_count)
        return fail(model_batch_errc::inconsistent_batch);
    if (batch.tokens.size() != batch.positions.size() || batch.kv_block_size != block_size) {
        return fail(model_batch_errc::inconsistent_batch);
    }

    paged_batch_metadata metadata;
    metadata.slots.reserve(batch.tokens.size());
    metadata.table_offsets.reserve(batch.tokens.size());
    metadata.table_lengths.reserve(batch.tokens.size());
    metadata.logits_indices.reserve(batch.items.size());

    std::size_t expected_token_offset = 0;
    constexpr auto max_u32 = std::numeric_limits<std::uint32_t>::max();
    for (const auto& item : batch.items) {
        if (item.token_count == 0
            || item.token_offset != expected_token_offset
            || item.token_count > batch.tokens.size() - expected_token_offset
            || (item.logits_index
                && (*item.logits_index < item.token_offset
                    || *item.logits_index >= item.token_offset + item.token_count))
            || item.block_table.empty()
            || metadata.block_table.size() > max_u32
            || item.block_table.size() > max_u32 - metadata.block_table.size()) {
            return fail(model_batch_errc::inconsistent_batch);
        }

        const auto table_offset = static_cast<std::uint32_t>(metadata.block_table.size());
        const auto table_length = static_cast<std::uint32_t>(item.block_table.size());
        for (const auto physical_block : item.block_table) {
            if (physical_block >= block_count) {
                return fail(model_batch_errc::inconsistent_batch);
            }
            metadata.block_table.push_back(physical_block);
        }

        for (std::size_t row = item.token_offset; row < item.token_offset + item.token_count;
             ++row) {
            const auto position = batch.positions[row];
            if (position >= max_positions) {
                return fail(model_batch_errc::inconsistent_batch);
            }
            const auto logical_block = static_cast<std::size_t>(position) / block_size;
            if (logical_block >= item.block_table.size()) {
                return fail(model_batch_errc::inconsistent_batch);
            }

            const auto physical_block = item.block_table[logical_block];
            const auto token_offset = static_cast<std::size_t>(position) % block_size;
            const auto slot = static_cast<std::size_t>(physical_block) * block_size + token_offset;
            if (slot > max_u32) {
                return fail(model_batch_errc::inconsistent_batch);
            }

            metadata.slots.push_back(static_cast<std::uint32_t>(slot));
            metadata.table_offsets.push_back(table_offset);
            metadata.table_lengths.push_back(table_length);
        }

        if (item.logits_index)
            metadata.logits_indices.push_back(*item.logits_index);
        expected_token_offset += item.token_count;
    }

    if (expected_token_offset != batch.tokens.size()) {
        return fail(model_batch_errc::inconsistent_batch);
    }
    return metadata;
}

} // namespace chibillm
