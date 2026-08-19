#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>
namespace chibillm {

using token_id = std::int32_t; // all vocabularies should fit into this
using seq_id = std::uint64_t;
using block_id = std::uint32_t; // KV-cache block identifier

// seq_status models the sequence status with the following statuses
// - waiting -> prompt or a recomputed pormpt after preemption work needs prefill work
// - running -> the prompt is complete and the sequence is ready for one-token decode steps.
// - finished -> EOS or max_tokens is reached
//
// The status can also go from running -> waiting if there is too much memory pressure.
enum class seq_status {
    waiting,
    running,
    finished,
};

enum class finish_reason {
    none,
    eos,
    len_limit,
    cancelled,
};

// errors that should be handled by the caller
enum class seq_errc : std::uint8_t {
    empty_prompt,
    invalid_temperature,
    invalid_max_new_tokens,
    invalid_block_size,
    already_finished,
    invalid_state_transition,
    invalid_finish_reason,
    zero_scheduled_tokens,
    work_already_scheduled,
    too_many_scheduled_tokens,
    no_work_scheduled,
    cached_token_count_out_of_range,
    index_out_of_range,
    block_table_full,
};

struct sampling_params {
    float temperature { 1.0f };
    std::size_t max_new_tokens { 64 };
    bool ignore_eos { false };
};

struct seq {
    [[nodiscard]]
    static std::expected<seq, seq_errc> make(
        seq_id id,
        std::vector<token_id> prompt_tokens,
        sampling_params params,
        std::size_t kv_block_size);

    seq(const seq&) = delete;
    seq& operator=(const seq&) = delete;

    seq(seq&&) noexcept = default;
    seq& operator=(seq&&) noexcept = default;

    [[nodiscard]] seq_id id() const noexcept;
    [[nodiscard]] seq_status status() const noexcept;
    [[nodiscard]] finish_reason reason() const noexcept;
    [[nodiscard]] bool is_finished() const noexcept;

    [[nodiscard]] std::span<const token_id> tokens() const noexcept;
    [[nodiscard]] std::span<const token_id> prompt_tokens() const noexcept;
    [[nodiscard]] std::span<const token_id> completion_tokens() const noexcept;

    [[nodiscard]] token_id last_token() const noexcept;

    [[nodiscard]] std::size_t token_count() const noexcept;
    [[nodiscard]] std::size_t prompt_token_count() const noexcept;
    [[nodiscard]] std::size_t completion_token_count() const noexcept;

    // Append a token sampled by the model.
    //
    // Requirements:
    // - status must be running;
    // - no model work may still be scheduled;
    // - all old tokens must already be cached.
    //
    // Important: this must not increment cached_token_count_. The sampled token
    // has not been processed by the transformer yet.
    [[nodiscard]]
    std::expected<void, seq_errc> append_token(token_id token);

    // Lifecycle transitions
    //
    // mark_running():
    //     waiting -> running
    //     Only valid after all current tokens have been cached.
    //
    // mark_waiting():
    //     running -> waiting
    //     Used when preempting a sequence.
    //
    // finish():
    //     waiting/running -> finished
    //     finish_reason::none is not a valid terminal reason.
    [[nodiscard]] std::expected<void, seq_errc> mark_running();
    [[nodiscard]] std::expected<void, seq_errc> mark_waiting();

    [[nodiscard]]
    std::expected<void, seq_errc> finish(finish_reason reason);

    // KV-cache progress
    [[nodiscard]] std::size_t cached_token_count() const noexcept;
    [[nodiscard]] std::size_t scheduled_token_count() const noexcept;

    // Number of tokens that do not yet have valid K/V entries. Scheduled tokens
    // still count as uncached until the model execution succeeds.
    [[nodiscard]] std::size_t uncached_token_count() const noexcept;

    // Number of additional tokens the scheduler may reserve:
    //
    // token_count - cached_token_count - scheduled_token_count
    [[nodiscard]] std::size_t schedulable_token_count() const noexcept;

    // Reserve uncached tokens for the next model invocation.
    //
    // Reject:
    // - zero tokens;
    // - a second reservation while work is already scheduled;
    // - more tokens than schedulable_token_count().
    [[nodiscard]]
    std::expected<void, seq_errc>
    schedule_tokens(std::size_t count);

    // Call only after successful model execution. Moves all scheduled tokens into
    // the cached prefix and resets scheduled_token_count_ to zero.
    [[nodiscard]]
    std::expected<void, seq_errc> commit_scheduled_tokens();

    // Use when backend execution failed before producing valid K/V entries.
    // This clears scheduled_token_count_ without advancing cached_token_count_.
    void cancel_scheduled_tokens() noexcept;

    // Used by the block manager when prefix-cache blocks are reused.
    // The count must not exceed token_count().
    [[nodiscard]]
    std::expected<void, seq_errc>
    set_cached_token_count(std::size_t count);

    // Block geometry
    [[nodiscard]] std::size_t block_size() const noexcept;

    // Number of logical token blocks required, including a partially full final
    // block.
    [[nodiscard]] std::size_t logical_block_count() const noexcept;

    // Number of valid tokens in the final logical block.
    //
    // For block size 16:
    // - 16 total tokens -> 16
    // - 17 total tokens -> 1
    // - 31 total tokens -> 15
    // - 32 total tokens -> 16
    [[nodiscard]] std::size_t tokens_in_last_block() const noexcept;

    // Return the token slice covered by one logical block.
    [[nodiscard]]
    std::expected<std::span<const token_id>, seq_errc>
    logical_block_tokens(std::size_t logical_block) const noexcept;

    // Logical block index -> physical KV-cache block ID.
    [[nodiscard]] std::span<const block_id> block_table() const noexcept;

    // Called by the future block manager when assigning physical storage.
    // Do not permit more physical blocks than logical_block_count().
    [[nodiscard]]
    std::expected<void, seq_errc>
    append_physical_block(block_id physical_block);

    // Called after the block manager has decremented all physical block reference
    // counts. The sequence must not be running or have scheduled work. Preserve
    // tokens but reset the cached count and physical block IDs.
    [[nodiscard]]
    std::expected<void, seq_errc> reset_cache_metadata();

    [[nodiscard]] const sampling_params& params() const noexcept;

    // Return:
    // - eos if the last generated token is EOS and ignore_eos is false;
    // - len_limit if completion_token_count() reached max_new_tokens;
    // - none otherwise.
    [[nodiscard]]
    finish_reason evaluate_stop(token_id eos_token) const noexcept;

private:
    seq(
        seq_id id,
        std::vector<token_id> prompt_tokens,
        sampling_params params,
        std::size_t kv_block_size);

    // Use assertions here to detect bugs inside your own implementation.
    void assert_invariants() const noexcept;

    seq_id id_;
    seq_status status_ { seq_status::waiting };
    finish_reason finish_reason_ { finish_reason::none };

    // Layout: [prompt tokens | completion tokens]
    std::vector<token_id> tokens_;

    // Immutable in meaning, even though it is not declared const. Keeping members
    // assignable makes the overall seq type easier to store in containers.
    std::size_t prompt_token_count_;

    // cached_token_count_ is always the length of a prefix of tokens_.
    std::size_t cached_token_count_ { 0 };
    std::size_t scheduled_token_count_ { 0 };

    std::size_t block_size_;
    std::vector<block_id> block_table_;

    sampling_params params_;
};
}
