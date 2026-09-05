#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "block_manager.h"
#include "result.h"
#include "seq.h"

namespace chibillm {

using batch_id = std::uint64_t;

// prefill processes prompt chunks; decode processes one token per sequence.
enum class batch_phase : std::uint8_t {
    prefill,
    decode,
};

struct scheduler_config {
    std::size_t max_sequences { 4 };
    std::size_t max_batch_tokens { 128 };
    std::size_t kv_block_count { 256 };
    std::size_t kv_block_size { 16 };
    token_id eos_token { 0 };
};

struct sequence_update {
    seq_id id;
    token_id token;
    finish_reason reason;
};

struct scheduled_item {
    seq_id id;
    std::size_t token_count;
    bool sample { true };
};

struct scheduled_batch {
    batch_id id;
    batch_phase phase;
    std::vector<scheduled_item> items;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t token_count() const noexcept;
};

enum class scheduler_errc : std::uint8_t {
    invalid_max_sequences,
    invalid_max_batch_tokens,
    invalid_kv_block_count,
    invalid_kv_block_size,

    duplicate_sequence_id,
    invalid_sequence_state,
    unknown_sequence,

    batch_in_flight,
    no_batch_in_flight,
    batch_id_mismatch,
    result_count_mismatch,
    no_runnable_sequences,
    cache_capacity_exhausted,

    block_manager_failure,
    sequence_failure,
};

// owns sequences and coordinates model work with cache ownership.
class scheduler {
public:
    [[nodiscard]] static result<scheduler, scheduler_errc>
    make(scheduler_config config, std::unique_ptr<model_state> state = {});

    scheduler(const scheduler&) = delete;
    scheduler& operator=(const scheduler&) = delete;
    scheduler(scheduler&&) noexcept = default;
    scheduler& operator=(scheduler&&) noexcept = default;

    [[nodiscard]] const scheduler_config& config() const noexcept;

    [[nodiscard]] std::size_t sequence_count() const noexcept;
    [[nodiscard]] std::size_t waiting_count() const noexcept;
    [[nodiscard]] std::size_t running_count() const noexcept;
    [[nodiscard]] bool has_in_flight_batch() const noexcept;

    // finished sequences remain available in sequences_.
    [[nodiscard]] bool is_finished() const noexcept;

    [[nodiscard]] const seq* find_sequence(seq_id id) const noexcept;

    [[nodiscard]] const model_state&
    state() const noexcept
    {
        return *state_;
    }

    [[nodiscard]] model_state&
    state() noexcept
    {
        return *state_;
    }

    // accepts a pristine waiting sequence.
    [[nodiscard]] result<void, scheduler_errc> add(seq sequence);

    // returns one prefill or decode reservation.
    [[nodiscard]] result<scheduled_batch, scheduler_errc> schedule();

    result<void, scheduler_errc> begin_execution(const model_batch& batch);

    // commits model work and applies samples only for completed prefill and decode items.
    [[nodiscard]] result<std::vector<sequence_update>, scheduler_errc>
    complete(batch_id id, std::span<const token_id> sampled_tokens);

    // cancels an in-flight reservation without releasing cache capacity.
    [[nodiscard]] result<void, scheduler_errc> abort(batch_id id);

    // Request lifecycle operations are valid only between model batches.
    [[nodiscard]] result<void, scheduler_errc> cancel(seq_id id);
    [[nodiscard]] result<void, scheduler_errc> remove(seq_id id);

private:
    seq* mutable_sequence(seq_id id) noexcept;
    scheduler(scheduler_config config, std::unique_ptr<model_state> state);

    static bool remove_from_queue(std::deque<seq_id>& queue, seq_id id) noexcept;

    // cancels token reservations but keeps allocated cache blocks.
    void rollback_reservations(const scheduled_batch& batch) noexcept;

    void assert_invariants() const noexcept;

    scheduler_config config_;
    std::unique_ptr<model_state> state_;

    // the map owns sequences; queues store ids.
    std::unordered_map<seq_id, seq> sequences_;
    std::deque<seq_id> waiting_;
    std::deque<seq_id> running_;

    // only one model invocation may be in flight.
    std::optional<scheduled_batch> active_batch_;
    bool state_transaction_open_ = false;
    batch_id next_batch_id_ { 1 };
};

} // namespace chibillm
