#include "scheduler.h"
#include "model_batch.h"
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
scheduler::make(scheduler_config config, std::unique_ptr<model_state> state)
{
    if (config.max_sequences == 0) {
        return fail(scheduler_errc::invalid_max_sequences);
    }

    if (config.max_batch_tokens == 0) {
        return fail(scheduler_errc::invalid_max_batch_tokens);
    }

    if (!state) {
        if (!config.kv_block_count)
            return fail(scheduler_errc::invalid_kv_block_count);
        if (!config.kv_block_size)
            return fail(scheduler_errc::invalid_kv_block_size);
        auto manager = block_manager::make(config.kv_block_count, config.kv_block_size);
        if (!manager)
            return fail(scheduler_errc::block_manager_failure);
        state = std::make_unique<block_manager>(std::move(*manager));
    }
    return scheduler { config, std::move(state) };
}

scheduler::scheduler(scheduler_config config, std::unique_ptr<model_state> state)
    : config_(config)
    , state_(std::move(state))
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
scheduler::mutable_sequence(seq_id id) noexcept
{
    const auto found = sequences_.find(id);
    return found == sequences_.end() ? nullptr : &found->second;
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
        && sequence.processed_token_count() == 0;

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

    bool blocked_by_cache = false;
    const auto select = [&](const std::deque<seq_id>& queue,
                            batch_phase phase) -> result<void, scheduler_errc> {
        batch.phase = phase;
        batch.items.reserve(std::min(queue.size(), config_.max_sequences));
        std::size_t used_tokens = 0;
        for (const auto id : queue) {
            if (batch.items.size() >= config_.max_sequences
                || used_tokens >= config_.max_batch_tokens)
                break;
            auto* sequence = mutable_sequence(id);
            assert(sequence != nullptr);
            const auto available = sequence->schedulable_token_count();
            if (!available || (phase == batch_phase::decode && available != 1))
                return fail(scheduler_errc::invalid_sequence_state);
            auto capacity = state_->reserve(id, sequence->token_count());
            if (!capacity) {
                if (capacity.error() == state_errc::capacity_exhausted) {
                    blocked_by_cache = true;
                    continue;
                }
                return fail(scheduler_errc::block_manager_failure);
            }
            const auto count = std::min(available, config_.max_batch_tokens - used_tokens);
            if (!sequence->schedule_tokens(count))
                return fail(scheduler_errc::sequence_failure);
            batch.items.push_back({ id, count, count == available });
            used_tokens += count;
        }
        return {};
    };
    auto selected = select(waiting_, batch_phase::prefill);
    if (selected && batch.empty())
        selected = select(running_, batch_phase::decode);
    if (!selected) {
        rollback_reservations(batch);
        return fail(selected.error());
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
scheduler::begin_execution(const model_batch& batch)
{
    if (!active_batch_)
        return fail(scheduler_errc::no_batch_in_flight);
    if (active_batch_->id != batch.id)
        return fail(scheduler_errc::batch_id_mismatch);
    if (state_transaction_open_)
        return fail(scheduler_errc::batch_in_flight);
    state_transaction_open_ = true;
    if (!state_->begin_batch(batch))
        return fail(scheduler_errc::block_manager_failure);
    return {};
}

result<std::vector<sequence_update>, scheduler_errc>
scheduler::complete(batch_id id, std::span<const token_id> sampled_tokens)
{
    if (!active_batch_.has_value()) {
        return fail(scheduler_errc::no_batch_in_flight);
    }

    const auto& active = *active_batch_;
    if (id != active.id) {
        return fail(scheduler_errc::batch_id_mismatch);
    }

    const auto sample_count = std::count_if(active.items.begin(), active.items.end(),
                                            [](const auto& item) { return item.sample; });
    if (sampled_tokens.size() != static_cast<std::size_t>(sample_count)) {
        return fail(scheduler_errc::result_count_mismatch);
    }
    std::vector<sequence_update> updates;
    updates.reserve(sampled_tokens.size());
    std::size_t sample_index = 0;

    if (state_transaction_open_) {
        state_->commit_batch();
        state_transaction_open_ = false;
    }
    for (std::size_t index = 0; index < active.items.size(); ++index) {
        const auto item = active.items[index];
        auto* sequence = mutable_sequence(item.id);
        assert(sequence != nullptr);

        auto committed = sequence->commit_scheduled_tokens();
        if (!committed) {
            assert(false && "prevalidated scheduled-token commit failed");
            return fail(scheduler_errc::sequence_failure);
        }

        if (!item.sample) {
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
        auto appended = sequence->append_token(sampled_tokens[sample_index++]);
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

            state_->release(item.id);
        } else if (active.phase == batch_phase::prefill) {
            if (!remove_from_queue(waiting_, item.id)) {
                assert(false && "prefilled sequence was absent from waiting queue");
                return fail(scheduler_errc::invalid_sequence_state);
            }
            running_.push_back(item.id);
        }
        updates.push_back({ item.id, sequence->last_token(), sequence->reason() });
    }

    active_batch_.reset();
    assert_invariants();

    return updates;
}

result<void, scheduler_errc>
scheduler::abort(batch_id id)
{
    if (!active_batch_.has_value()) {
        return fail(scheduler_errc::no_batch_in_flight);
    }

    const auto& active = *active_batch_;
    if (id != active.id) {
        return fail(scheduler_errc::batch_id_mismatch);
    }

    if (state_transaction_open_) {
        state_->abort_batch();
        state_transaction_open_ = false;
    }
    rollback_reservations(active);
    active_batch_.reset();
    assert_invariants();

    return {};
}

result<void, scheduler_errc>
scheduler::cancel(seq_id id)
{
    if (active_batch_.has_value()) {
        return fail(scheduler_errc::batch_in_flight);
    }

    auto* sequence = mutable_sequence(id);
    if (sequence == nullptr) {
        return fail(scheduler_errc::unknown_sequence);
    }
    if (sequence->is_finished()) {
        return {};
    }

    auto& queue = sequence->status() == seq_status::waiting ? waiting_ : running_;
    if (!remove_from_queue(queue, id)) {
        return fail(scheduler_errc::invalid_sequence_state);
    }
    auto finished = sequence->finish(finish_reason::cancelled);
    if (!finished) {
        return fail(scheduler_errc::sequence_failure);
    }
    state_->release(id);

    assert_invariants();
    return {};
}

result<void, scheduler_errc>
scheduler::remove(seq_id id)
{
    if (active_batch_.has_value()) {
        return fail(scheduler_errc::batch_in_flight);
    }

    const auto found = sequences_.find(id);
    if (found == sequences_.end()) {
        return fail(scheduler_errc::unknown_sequence);
    }
    if (!found->second.is_finished()) {
        return fail(scheduler_errc::invalid_sequence_state);
    }

    sequences_.erase(found);
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
        auto* sequence = mutable_sequence(item.id);

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
    assert(state_ != nullptr);
#endif
}

} // namespace chibillm
