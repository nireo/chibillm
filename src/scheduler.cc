#include "scheduler.h"
#include "model_batch.h"
#include "seq.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <new>
#include <utility>

namespace chibillm {
namespace {
std::optional<std::size_t>
cache_budget(const seq& sequence, std::size_t block_size) noexcept
{
    const auto prompt = sequence.prompt_token_count();
    // The terminal sample finishes the sequence without another forward pass.
    const auto output = sequence.params().max_new_tokens - 1;
    if (output > std::numeric_limits<std::size_t>::max() - prompt)
        return std::nullopt;
    return 1 + (prompt + output - 1) / block_size;
}
} // namespace

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

result<scheduler, scheduler_error>
scheduler::make(scheduler_config config, token_id eos_token, std::unique_ptr<model_state> state)
{
    if (config.max_sequences == 0) {
        return fail(scheduler_errc::invalid_max_sequences);
    }

    if (config.max_batch_tokens == 0) {
        return fail(scheduler_errc::invalid_max_batch_tokens);
    }

    if (!config.prefill_chunk_tokens)
        return fail(scheduler_errc::invalid_prefill_chunk_tokens);

    if (!state) {
        if (!config.kv_block_count)
            return fail(scheduler_errc::invalid_kv_block_count);
        if (!config.kv_block_size)
            return fail(scheduler_errc::invalid_kv_block_size);
        auto manager = block_manager::make(config.kv_block_count, config.kv_block_size);
        if (!manager)
            return fail(scheduler_errc::block_manager_failure, manager.error());
        state = std::make_unique<block_manager>(std::move(*manager));
    }
    if (state->block_size()) {
        if (!config.kv_block_count)
            return fail(scheduler_errc::invalid_kv_block_count);
        if (config.kv_block_size != state->block_size())
            return fail(scheduler_errc::invalid_kv_block_size);
    }
    return scheduler { config, eos_token, std::move(state) };
}

scheduler::scheduler(scheduler_config config,
                     token_id eos_token,
                     std::unique_ptr<model_state> state)
    : config_(config)
    , eos_token_(eos_token)
    , state_(std::move(state))
    , prefixes_(config.prefix_cache_bytes)
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

result<void, scheduler_error>
scheduler::add(seq sequence)
{
    auto admitted = try_add(sequence);
    if (!admitted)
        return fail(admitted.error());
    switch (*admitted) {
    case admission_result::admitted:
        return {};
    case admission_result::deferred:
        return fail(scheduler_errc::admission_deferred);
    case admission_result::too_large:
        return fail(scheduler_errc::sequence_exceeds_cache_capacity);
    }
    std::unreachable();
}

bool
scheduler::fits_cache(const seq& sequence) const noexcept
{
    if (!state_->block_size())
        return true;
    const auto blocks = cache_budget(sequence, state_->block_size());
    return blocks && *blocks <= config_.kv_block_count;
}

result<admission_result, scheduler_error>
scheduler::try_add(seq& sequence)
{
    if (active_batch_)
        return fail(scheduler_errc::batch_in_flight);
    const auto id = sequence.id();

    if (sequences_.contains(id)) {
        return fail(scheduler_errc::duplicate_sequence_id);
    }

    const bool is_ok = sequence.status() == seq_status::waiting
        && sequence.scheduled_token_count() == 0
        && sequence.processed_token_count() == 0
        && sequence.token_count() == sequence.prompt_token_count();

    if (!is_ok) {
        return fail(scheduler_errc::invalid_sequence_state);
    }

    if (!fits_cache(sequence))
        return admission_result::too_large;
    if (waiting_.size() + running_.size() >= config_.max_sequences)
        return admission_result::deferred;

    if (state_->block_size()) {
        auto available = config_.kv_block_count;
        for (const auto& queue : { &waiting_, &running_ }) {
            for (const auto active : *queue) {
                const auto blocks = cache_budget(*find_sequence(active), state_->block_size());
                assert(blocks && *blocks <= available);
                available -= *blocks;
            }
        }
        if (*cache_budget(sequence, state_->block_size()) > available)
            return admission_result::deferred;
    }

    // Create sequence-local state here so allocation failures affect only this admission.
    auto reserved = state_->reserve(id, 1);
    if (!reserved) {
        state_->release(id);
        return fail(scheduler_errc::block_manager_failure, reserved.error(), "admit sequence");
    }

    if (const auto* cached = prefixes_.find(sequence.prompt_tokens())) {
        bool restored = false;
        try {
            restored = state_->restore(id, *cached).has_value();
        } catch (const std::bad_alloc&) {}
        if (restored) {
            auto applied = sequence.restore_prefix(cached->token_count);
            assert(applied);
        } else {
            // A cache miss must not prevent otherwise valid admission.
            state_->release(id);
            reserved = state_->reserve(id, 1);
            if (!reserved) {
                state_->release(id);
                return fail(scheduler_errc::block_manager_failure, reserved.error(),
                            "admit sequence");
            }
        }
    }

    auto insert = sequences_.try_emplace(id, std::move(sequence));
    if (!insert.second) {
        return fail(scheduler_errc::duplicate_sequence_id);
    }

    waiting_.push_back(id);
    assert_invariants();
    return admission_result::admitted;
}

result<scheduled_batch, scheduler_error>
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

    if (waiting_.empty() && running_.empty())
        return fail(scheduler_errc::no_runnable_sequences);
    if (std::holds_alternative<prefill_turn>(turn_) && waiting_.empty())
        turn_ = decode_round { running_.size() };

    const auto* round = std::get_if<decode_round>(&turn_);
    batch.phase = round ? batch_phase::decode : batch_phase::prefill;
    const auto& queue = round ? running_ : waiting_;
    const auto item_limit = round ? round->remaining : queue.size();
    const bool contended = waiting_.size() > 1 || !running_.empty();
    const auto quantum = batch.phase == batch_phase::prefill && contended
        ? config_.prefill_chunk_tokens
        : config_.max_batch_tokens;
    const auto select = [&]() -> result<void, scheduler_error> {
        batch.items.reserve(std::min(item_limit, config_.max_sequences));
        std::size_t used_tokens = 0;
        for (const auto id : queue) {
            if (batch.items.size() >= std::min(item_limit, config_.max_sequences)
                || used_tokens >= config_.max_batch_tokens)
                break;
            auto* sequence = mutable_sequence(id);
            assert(sequence != nullptr);
            const auto available = sequence->schedulable_token_count();
            if (!available || (batch.phase == batch_phase::decode && available != 1))
                return fail(scheduler_errc::invalid_sequence_state);
            auto count = std::min({ available, quantum, config_.max_batch_tokens - used_tokens });
            if (batch.phase == batch_phase::prefill) {
                const auto start = sequence->processed_token_count();
                for (const auto boundary : sequence->checkpoints()) {
                    if (boundary > start
                        && boundary < start + count
                        && cacheable_prefix(boundary)
                        && !prefixes_.contains(sequence->prompt_tokens().first(boundary))) {
                        count = boundary - start;
                        break;
                    }
                }
            }
            auto capacity = state_->reserve(id, sequence->processed_token_count() + count);
            if (!capacity) {
                if (capacity.error() == state_errc::capacity_exhausted) {
                    return fail(scheduler_errc::cache_capacity_exhausted, capacity.error());
                }
                return fail(scheduler_errc::block_manager_failure, capacity.error(),
                            "reserve sequence " + std::to_string(id));
            }
            auto reserved = sequence->schedule_tokens(count);
            if (!reserved)
                return fail(scheduler_errc::sequence_failure, reserved.error(), "reserve tokens");
            batch.items.push_back({ id, count, count == available });
            used_tokens += count;
        }
        return {};
    };
    auto selected = select();
    if (!selected) {
        rollback_reservations(batch);
        return fail(selected.error());
    }

    if (batch.empty()) {
        return fail(scheduler_errc::no_runnable_sequences);
    }

    active_batch_ = batch;
    ++next_batch_id_;

    assert_invariants();
    return batch;
}

result<void, scheduler_error>
scheduler::begin_execution(const model_batch& batch)
{
    if (!active_batch_)
        return fail(scheduler_errc::no_batch_in_flight);
    if (active_batch_->id != batch.id)
        return fail(scheduler_errc::batch_id_mismatch);
    if (state_transaction_open_)
        return fail(scheduler_errc::batch_in_flight);
    state_transaction_open_ = true;
    auto begun = state_->begin_batch(batch);
    if (!begun)
        return fail(scheduler_errc::block_manager_failure, begun.error(),
                    "begin model state transaction");
    return {};
}

result<std::vector<sequence_update>, scheduler_error>
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
            return fail(scheduler_errc::sequence_failure, committed.error());
        }
        if (active.phase == batch_phase::prefill)
            cache_prefix(*sequence);

        auto& source = active.phase == batch_phase::prefill ? waiting_ : running_;
        assert(!source.empty() && source.front() == item.id);
        source.pop_front();

        if (!item.sample) {
            waiting_.push_back(item.id);
            continue;
        }

        if (active.phase == batch_phase::prefill) {
            auto running = sequence->mark_running();
            if (!running) {
                assert(false && "prevalidated transition to running failed");
                return fail(scheduler_errc::sequence_failure, running.error());
            }
        }

        // appending the sample restores the one-token cache gap.
        auto appended = sequence->append_token(sampled_tokens[sample_index++]);
        if (!appended) {
            assert(false && "prevalidated sampled-token append failed");
            return fail(scheduler_errc::sequence_failure, appended.error());
        }

        const auto stop_reason = sequence->evaluate_stop(eos_token_);
        if (stop_reason != finish_reason::none) {
            auto finished = sequence->finish(stop_reason);
            if (!finished) {
                assert(false && "prevalidated sequence finish failed");
                return fail(scheduler_errc::sequence_failure, finished.error());
            }

            state_->release(item.id);
        } else {
            running_.push_back(item.id);
        }
        updates.push_back({ item.id, sequence->last_token(), sequence->reason() });
    }

    if (active.phase == batch_phase::prefill) {
        if (!running_.empty())
            turn_ = decode_round { running_.size() };
    } else {
        auto& round = std::get<decode_round>(turn_);
        round.remaining -= active.items.size();
        if (!round.remaining)
            turn_ = prefill_turn {};
    }

    active_batch_.reset();
    assert_invariants();

    return updates;
}

result<void, scheduler_error>
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

result<void, scheduler_error>
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
    if (sequence->status() == seq_status::running) {
        if (auto* round = std::get_if<decode_round>(&turn_)) {
            const auto position = std::find(running_.begin(), running_.end(), id);
            if (static_cast<std::size_t>(position - running_.begin()) < round->remaining) {
                if (!--round->remaining)
                    turn_ = prefill_turn {};
            }
        }
    }
    if (!remove_from_queue(queue, id)) {
        return fail(scheduler_errc::invalid_sequence_state);
    }
    auto finished = sequence->finish(finish_reason::cancelled);
    if (!finished) {
        return fail(scheduler_errc::sequence_failure, finished.error());
    }
    state_->release(id);

    assert_invariants();
    return {};
}

result<void, scheduler_error>
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
scheduler::cache_prefix(const seq& sequence) noexcept
try {
    const auto count = sequence.processed_token_count();
    if (!std::ranges::binary_search(sequence.checkpoints(), count) || !cacheable_prefix(count))
        return;
    const auto tokens = sequence.prompt_tokens().first(count);
    if (prefixes_.contains(tokens))
        return;
    prefixes_.make_room(count, state_->checkpoint_bytes(count));
    auto checkpoint = state_->checkpoint(sequence.id(), count);
    if (checkpoint)
        prefixes_.insert(tokens, std::move(*checkpoint));
} catch (const std::bad_alloc&) {
    // Checkpoint allocation is optional; generation can continue without it.
}

bool
scheduler::cacheable_prefix(std::size_t count) const noexcept
{
    return count >= config_.prefix_cache_min_tokens
        && prefixes_.fits(count, state_->checkpoint_bytes(count));
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
    assert(config_.prefill_chunk_tokens > 0);
    assert(state_ != nullptr);
    assert(waiting_.size() + running_.size() <= config_.max_sequences);
    if (const auto* round = std::get_if<decode_round>(&turn_))
        assert(round->remaining > 0 && round->remaining <= running_.size());
#endif
}

} // namespace chibillm
