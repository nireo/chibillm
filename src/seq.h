#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "result.h"

namespace chibillm {

using token_id = std::int32_t;
using seq_id = std::uint64_t;
using block_id = std::uint32_t;

// waiting requires prefill; running is ready for one-token decode.
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
    [[nodiscard]] static result<seq, seq_errc> make(seq_id id,
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

    // appends an unprocessed model sample and preserves the one-token cache gap.
    [[nodiscard]] result<void, seq_errc> append_token(token_id token);

    [[nodiscard]] result<void, seq_errc> mark_running();
    [[nodiscard]] result<void, seq_errc> mark_waiting();
    [[nodiscard]] result<void, seq_errc> finish(finish_reason reason);

    [[nodiscard]] std::size_t cached_token_count() const noexcept;
    [[nodiscard]] std::size_t scheduled_token_count() const noexcept;

    // scheduled tokens stay uncached until model execution succeeds.
    [[nodiscard]] std::size_t uncached_token_count() const noexcept;

    // token_count - cached_token_count - scheduled_token_count
    [[nodiscard]] std::size_t schedulable_token_count() const noexcept;

    [[nodiscard]] result<void, seq_errc> schedule_tokens(std::size_t count);
    [[nodiscard]] result<void, seq_errc> commit_scheduled_tokens();
    void cancel_scheduled_tokens() noexcept;

    [[nodiscard]] result<void, seq_errc> set_cached_token_count(std::size_t count);

    [[nodiscard]] std::size_t block_size() const noexcept;
    [[nodiscard]] std::size_t logical_block_count() const noexcept;
    [[nodiscard]] std::size_t tokens_in_last_block() const noexcept;

    [[nodiscard]] result<std::span<const token_id>, seq_errc>
    logical_block_tokens(std::size_t logical_block) const noexcept;

    // maps logical block indices to physical cache blocks.
    [[nodiscard]] std::span<const block_id> block_table() const noexcept;

    [[nodiscard]] result<void, seq_errc> append_physical_block(block_id physical_block);

    // clears cache metadata without removing tokens.
    [[nodiscard]] result<void, seq_errc> reset_cache_metadata();

    [[nodiscard]] const sampling_params& params() const noexcept;

    // eos takes precedence over the length limit.
    [[nodiscard]] finish_reason evaluate_stop(token_id eos_token) const noexcept;

private:
    seq(seq_id id,
        std::vector<token_id> prompt_tokens,
        sampling_params params,
        std::size_t kv_block_size);

    void assert_invariants() const noexcept;

    seq_id id_;
    seq_status status_ { seq_status::waiting };
    finish_reason finish_reason_ { finish_reason::none };

    // layout: [prompt tokens | completion tokens]
    std::vector<token_id> tokens_;
    std::size_t prompt_token_count_;

    // cached tokens always form a prefix of tokens_.
    std::size_t cached_token_count_ { 0 };
    std::size_t scheduled_token_count_ { 0 };

    std::size_t block_size_;
    std::vector<block_id> block_table_;
    sampling_params params_;
};

} // namespace chibillm
