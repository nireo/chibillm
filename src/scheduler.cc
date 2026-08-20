#include "scheduler.h"
#include "seq.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace chibillm {

bool
scheduled_batch::empty() const noexcept
{
    return items.empty();
}

std::size_t
scheduled_batch::token_count() const noexcept
{
    std::size_t total = 0;
    for (const auto& item : items) {
        total += item.token_count;
    }

    return total;
}

result<scheduler, scheduler_errc>
scheduler::make(scheduler_config config)
{
    if (config.max_sequences == 0) {
        return fail(scheduler_errc::invalid_max_sequences);
    }

    if (config.max_batch_tokens == 0) {
        return fail(scheduler_errc::invalid_max_batch_tokens);
    }

    if (config.kv_block_count == 0) {
        return fail(scheduler_errc::invalid_kv_block_count);
    }

    if (config.kv_block_size == 0) {
        return fail(scheduler_errc::invalid_kv_block_size);
    }

    auto manager = block_manager::make(config.kv_block_count, config.kv_block_size);
    if (!manager) {
        // map block-manager failures at the scheduler boundary.
        return fail(scheduler_errc::block_manager_failure);
    }

    return scheduler { config, std::move(*manager) };
}

scheduler::scheduler(scheduler_config config, block_manager manager)
    : config_(config)
    , block_manager_(std::move(manager))
{
    assert_invariants();
}

const scheduler_config&
scheduler::config() const noexcept
{
    return config_;
}

std::size_t
scheduler::sequence_count() const noexcept
{
    return sequences_.size();
}

std::size_t
scheduler::waiting_count() const noexcept
{
    return waiting_.size();
}

std::size_t
scheduler::running_count() const noexcept
{
    return running_.size();
}

bool
scheduler::has_in_flight_batch() const noexcept
{
    return active_batch_.has_value();
}

bool
scheduler::is_finished() const noexcept
{
    return waiting_.empty() && running_.empty() && !active_batch_.has_value();
}

const seq*
scheduler::find_sequence(seq_id id) const noexcept
{
    const auto found = sequences_.find(id);
    return found == sequences_.end() ? nullptr : &found->second;
}

seq*
scheduler::find_sequence(seq_id id) noexcept
{
    const auto found = sequences_.find(id);
    return found == sequences_.end() ? nullptr : &found->second;
}

const block_manager&
scheduler::cache() const noexcept
{
    return block_manager_;
}

result<void, scheduler_errc>
scheduler::add(seq sequence)
{
    const auto id = sequence.id();

    if (sequences_.contains(id)) {
        return fail(scheduler_errc::duplicate_sequence_id);
    }

    const bool is_ok = sequence.status() == seq_status::waiting
        && sequence.scheduled_token_count() == 0
        && sequence.cached_token_count() == 0
        && sequence.block_table().empty()
        && sequence.block_size() == config_.kv_block_size;

    if (!is_ok) {
        return fail(scheduler_errc::invalid_sequence_state);
    }

    auto insert = sequences_.try_emplace(id, std::move(sequence));
    if (!insert.second) {
        return fail(scheduler_errc::duplicate_sequence_id);
    }

    waiting_.push_back(id);
    assert_invariants();
    return {};
}

result<scheduled_batch, scheduler_errc>
scheduler::schedule()
{
    if (active_batch_.has_value()) {
        return fail(scheduler_errc::batch_in_flight);
    }

    scheduled_batch batch {
        .id = next_batch_id_,
        .phase = batch_phase::prefill,
        .items = {},
    };

    batch.items.reserve(std::min(waiting_.size(), config_.max_sequences));
    size_t used_tokens = 0;
    bool blocked_by_cache = false;

    for (const auto id : waiting_) {
        if (batch.items.size() >= config_.max_sequences) {
            break;
        }

        if (used_tokens >= config_.max_batch_tokens) {
            break;
        }

        auto* sequence = find_sequence(id);
        if (sequence == nullptr) {
            rollback_reservations(batch);
            return fail(scheduler_errc::unknown_sequence);
        }

        const auto available = sequence->schedulable_token_count();
        if (available == 0) {
            rollback_reservations(batch);
            return fail(scheduler_errc::invalid_sequence_state);
        }

        const auto remaining_tokens = config_.max_batch_tokens - used_tokens;
        const auto tokens_to_schedule = std::min(available, remaining_tokens);

        // capacity covers the full sequence, not only this prefill chunk.
        auto capacity = block_manager_.ensure_capacity(*sequence);
        if (!capacity) {
            if (capacity.error() == block_manager_errc::insufficient_free_blocks) {
                blocked_by_cache = true;
                continue;
            }

            rollback_reservations(batch);
            return fail(scheduler_errc::block_manager_failure);
        }

        auto reserved = sequence->schedule_tokens(tokens_to_schedule);
        if (!reserved) {
            rollback_reservations(batch);
            return fail(scheduler_errc::sequence_failure);
        }

        batch.items.push_back(scheduled_item {
            .id = id,
            .token_count = tokens_to_schedule,
        });

        used_tokens += tokens_to_schedule;
    }

    // fall back to decode only when no prefill work fits.
    if (batch.empty()) {
        batch.phase = batch_phase::decode;
        batch.items.reserve(std::min(running_.size(), config_.max_sequences));
        used_tokens = 0;

        for (const auto id : running_) {
            if (batch.items.size() >= config_.max_sequences
                || used_tokens >= config_.max_batch_tokens) {
                break;
            }

            auto* sequence = find_sequence(id);
            if (sequence == nullptr) {
                rollback_reservations(batch);
                return fail(scheduler_errc::unknown_sequence);
            }

            // running sequences must preserve a one-token cache gap.
            if (sequence->status() != seq_status::running
                || sequence->schedulable_token_count() != 1) {
                rollback_reservations(batch);
                return fail(scheduler_errc::invalid_sequence_state);
            }

            auto capacity = block_manager_.ensure_capacity(*sequence);
            if (!capacity) {
                if (capacity.error() == block_manager_errc::insufficient_free_blocks) {
                    blocked_by_cache = true;
                    continue;
                }

                rollback_reservations(batch);
                return fail(scheduler_errc::block_manager_failure);
            }

            auto reserved = sequence->schedule_tokens(1);
            if (!reserved) {
                rollback_reservations(batch);
                return fail(scheduler_errc::sequence_failure);
            }

            batch.items.push_back(scheduled_item {
                .id = id,
                .token_count = 1,
            });
            ++used_tokens;
        }
    }

    if (batch.empty()) {
        if (blocked_by_cache) {
            return fail(scheduler_errc::cache_capacity_exhausted);
        }
        return fail(scheduler_errc::no_runnable_sequences);
    }

    active_batch_ = batch;
    ++next_batch_id_;

    assert_invariants();
    return batch;
}

result<void, scheduler_errc>
scheduler::complete(const scheduled_batch& batch, std::span<const token_id> sampled_tokens)
{
    if (!active_batch_.has_value()) {
        return fail(scheduler_errc::no_batch_in_flight);
    }

    const auto& active = *active_batch_;
    if (batch.id != active.id) {
        return fail(scheduler_errc::batch_id_mismatch);
    }

    if (sampled_tokens.size() != active.items.size()) {
        return fail(scheduler_errc::result_count_mismatch);
    }

    // validate the caller's copy against the authoritative batch.
    if (batch.phase != active.phase || batch.items.size() != active.items.size()) {
        return fail(scheduler_errc::invalid_sequence_state);
    }

    for (std::size_t index = 0; index < active.items.size(); ++index) {
        const auto& expected_item = active.items[index];
        const auto& supplied_item = batch.items[index];
        if (supplied_item.id != expected_item.id
            || supplied_item.token_count != expected_item.token_count) {
            return fail(scheduler_errc::invalid_sequence_state);
        }

        const auto* sequence = find_sequence(expected_item.id);
        if (sequence == nullptr) {
            return fail(scheduler_errc::unknown_sequence);
        }

        const auto expected_status =
            active.phase == batch_phase::prefill ? seq_status::waiting : seq_status::running;

        const auto& expected_queue = active.phase == batch_phase::prefill ? waiting_ : running_;

        if (sequence->status() != expected_status
            || sequence->scheduled_token_count() != expected_item.token_count
            || std::find(expected_queue.begin(), expected_queue.end(), expected_item.id)
                == expected_queue.end()) {
            return fail(scheduler_errc::invalid_sequence_state);
        }
    }

    // validate every item before mutating any sequence.
    for (std::size_t index = 0; index < active.items.size(); ++index) {
        const auto item = active.items[index];
        auto* sequence = find_sequence(item.id);
        assert(sequence != nullptr);

        auto committed = sequence->commit_scheduled_tokens();
        if (!committed) {
            assert(false && "prevalidated scheduled-token commit failed");
            return fail(scheduler_errc::sequence_failure);
        }

        if (active.phase == batch_phase::prefill && sequence->uncached_token_count() != 0) {
            // intermediate prefill samples are not usable completions.
            continue;
        }

        if (active.phase == batch_phase::prefill) {
            auto running = sequence->mark_running();
            if (!running) {
                assert(false && "prevalidated transition to running failed");
                return fail(scheduler_errc::sequence_failure);
            }
        }

        // appending the sample restores the one-token cache gap.
        auto appended = sequence->append_token(sampled_tokens[index]);
        if (!appended) {
            assert(false && "prevalidated sampled-token append failed");
            return fail(scheduler_errc::sequence_failure);
        }

        const auto stop_reason = sequence->evaluate_stop(config_.eos_token);
        if (stop_reason != finish_reason::none) {
            auto finished = sequence->finish(stop_reason);
            if (!finished) {
                assert(false && "prevalidated sequence finish failed");
                return fail(scheduler_errc::sequence_failure);
            }

            auto& source_queue = active.phase == batch_phase::prefill ? waiting_ : running_;
            if (!remove_from_queue(source_queue, item.id)) {
                assert(false && "completed sequence was absent from its queue");
                return fail(scheduler_errc::invalid_sequence_state);
            }

            auto released = block_manager_.release(*sequence);
            if (!released) {
                assert(false && "prevalidated KV-cache release failed");
                return fail(scheduler_errc::block_manager_failure);
            }
        } else if (active.phase == batch_phase::prefill) {
            if (!remove_from_queue(waiting_, item.id)) {
                assert(false && "prefilled sequence was absent from waiting queue");
                return fail(scheduler_errc::invalid_sequence_state);
            }
            running_.push_back(item.id);
        }
    }

    active_batch_.reset();
    assert_invariants();

    return {};
}

bool
scheduler::remove_from_queue(std::deque<seq_id>& queue, seq_id id) noexcept
{
    const auto found = std::find(queue.begin(), queue.end(), id);
    if (found == queue.end()) {
        return false;
    }

    queue.erase(found);
    return true;
}

void
scheduler::rollback_reservations(const scheduled_batch& batch) noexcept
{
    for (const auto& item : batch.items) {
        auto* sequence = find_sequence(item.id);

        assert(sequence != nullptr);
        if (sequence == nullptr) {
            continue;
        }

        assert(sequence->scheduled_token_count() == item.token_count);
        sequence->cancel_scheduled_tokens();
    }
}

void
scheduler::assert_invariants() const noexcept
{
#ifndef NDEBUG
    assert(config_.max_sequences > 0);
    assert(config_.max_batch_tokens > 0);
    assert(config_.kv_block_count > 0);
    assert(config_.kv_block_size > 0);
#endif
}

} // namespace chibillm
