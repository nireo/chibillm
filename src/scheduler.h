#pragma once

#include "error.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <variant>
#include <vector>

#include "block_manager.h"
#include "prefix_cache.h"
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
    // Applied only when other sequences compete for inference time.
    std::size_t prefill_chunk_tokens { 64 };
    std::size_t prefix_cache_bytes { 128 * 1024 * 1024 };
    std::size_t prefix_cache_min_tokens { 64 };
};

enum class admission_result : std::uint8_t {
    admitted,
    deferred,
    too_large,
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
    invalid_prefill_chunk_tokens,
    admission_deferred,
    sequence_exceeds_cache_capacity,
};

[[nodiscard]] inline std::string_view
error_name(scheduler_errc code) noexcept
{
    static constexpr std::array names {
        "scheduler.invalid_max_sequences",    "scheduler.invalid_max_batch_tokens",
        "scheduler.invalid_kv_block_count",   "scheduler.invalid_kv_block_size",
        "scheduler.duplicate_sequence_id",    "scheduler.invalid_sequence_state",
        "scheduler.unknown_sequence",         "scheduler.batch_in_flight",
        "scheduler.no_batch_in_flight",       "scheduler.batch_id_mismatch",
        "scheduler.result_count_mismatch",    "scheduler.no_runnable_sequences",
        "scheduler.cache_capacity_exhausted", "scheduler.block_manager_failure",
        "scheduler.sequence_failure",         "scheduler.invalid_prefill_chunk_tokens",
        "scheduler.admission_deferred",       "scheduler.sequence_exceeds_cache_capacity",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "scheduler.unknown_error";
}

using scheduler_error = error<scheduler_errc>;

// owns sequences and coordinates model work with cache ownership.
class scheduler {
public:
    [[nodiscard]] static result<scheduler, scheduler_error>
    make(scheduler_config config, token_id eos_token, std::unique_ptr<model_state> state = {});

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

    [[nodiscard]] bool fits_cache(const seq& sequence) const noexcept;

    // Only admitted moves the pristine sequence; pressure leaves it with the caller.
    // Admission, like cancellation, is valid only between model batches.
    [[nodiscard]] result<admission_result, scheduler_error> try_add(seq& sequence);

    // Strict admission for callers that do not maintain a pending queue.
    [[nodiscard]] result<void, scheduler_error> add(seq sequence);

    // returns one prefill or decode reservation.
    [[nodiscard]] result<scheduled_batch, scheduler_error> schedule();

    result<void, scheduler_error> begin_execution(const model_batch& batch);

    // commits model work and applies samples only for completed prefill and decode items.
    [[nodiscard]] result<std::vector<sequence_update>, scheduler_error>
    complete(batch_id id, std::span<const token_id> sampled_tokens);

    // cancels an in-flight reservation without releasing cache capacity.
    [[nodiscard]] result<void, scheduler_error> abort(batch_id id);

    // Request lifecycle operations are valid only between model batches.
    [[nodiscard]] result<void, scheduler_error> cancel(seq_id id);
    [[nodiscard]] result<void, scheduler_error> remove(seq_id id);

private:
    seq* mutable_sequence(seq_id id) noexcept;
    scheduler(scheduler_config config, token_id eos_token, std::unique_ptr<model_state> state);

    static bool remove_from_queue(std::deque<seq_id>& queue, seq_id id) noexcept;

    // cancels token reservations but keeps allocated cache blocks.
    void rollback_reservations(const scheduled_batch& batch) noexcept;
    bool cacheable_prefix(std::size_t count) const noexcept;
    void cache_prefix(const seq& sequence) noexcept;

    void assert_invariants() const noexcept;

    // Unvisited members of a decode round are the first remaining entries in running_.
    struct prefill_turn {};

    struct decode_round {
        std::size_t remaining;
    };

    std::variant<prefill_turn, decode_round> turn_ { prefill_turn {} };

    scheduler_config config_;
    token_id eos_token_;
    std::unique_ptr<model_state> state_;
    prefix_cache prefixes_;

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
