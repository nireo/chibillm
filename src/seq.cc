#include "seq.h"

#include <cassert>
#include <cmath>
#include <utility>

namespace chibillm {
seq::seq(
    seq_id id,
    std::vector<token_id> prompt_tokens,
    sampling_params params,
    std::size_t kv_block_size)
    : id_(id)
    , tokens_(std::move(prompt_tokens))
    , prompt_token_count_(tokens_.size())
    , block_size_(kv_block_size)
    , params_(params)
{
    assert_invariants();
}

std::expected<seq, seq_errc> seq::make(
    seq_id id,
    std::vector<token_id> prompt_tokens,
    sampling_params params,
    std::size_t kv_block_size)
{
    if (prompt_tokens.empty()) {
        return std::unexpected(seq_errc::empty_prompt);
    }

    if (!std::isfinite(params.temperature) || params.temperature <= 0.0f) {
        return std::unexpected(seq_errc::invalid_temperature);
    }

    if (params.max_new_tokens == 0) {
        return std::unexpected(seq_errc::invalid_max_new_tokens);
    }

    if (kv_block_size == 0) {
        return std::unexpected(seq_errc::invalid_block_size);
    }

    return seq {
        id,
        std::move(prompt_tokens),
        params,
        kv_block_size,
    };
}

seq_id seq::id() const noexcept
{
    return id_;
}

seq_status seq::status() const noexcept
{
    return status_;
}

finish_reason seq::reason() const noexcept
{
    return finish_reason_;
}

bool seq::is_finished() const noexcept
{
    return status_ == seq_status::finished;
}

std::span<const token_id> seq::tokens() const noexcept
{
    return {
        tokens_.data(),
        tokens_.size(),
    };
}

std::span<const token_id> seq::prompt_tokens() const noexcept
{
    return {
        tokens_.data(),
        prompt_token_count_,
    };
}

std::span<const token_id> seq::completion_tokens() const noexcept
{
    return {
        tokens_.data() + prompt_token_count_,
        tokens_.size() - prompt_token_count_,
    };
}

token_id seq::last_token() const noexcept
{
    // make() rejects empty prompts, and this class never removes tokens.
    return tokens_.back();
}

std::size_t seq::token_count() const noexcept
{
    return tokens_.size();
}

std::size_t seq::prompt_token_count() const noexcept
{
    return prompt_token_count_;
}

std::size_t seq::completion_token_count() const noexcept
{
    return tokens_.size() - prompt_token_count_;
}

std::expected<void, seq_errc> seq::append_token(token_id token)
{
    if (status_ == seq_status::finished) {
        return std::unexpected(seq_errc::already_finished);
    }

    if (status_ != seq_status::running) {
        return std::unexpected(seq_errc::invalid_state_transition);
    }

    if (scheduled_token_count_ != 0) {
        return std::unexpected(seq_errc::work_already_scheduled);
    }

    if (cached_token_count_ != token_count()) {
        return std::unexpected(seq_errc::invalid_state_transition);
    }

    tokens_.push_back(token);
    assert_invariants();
    return { };
}

std::expected<void, seq_errc> seq::mark_running()
{
    if (status_ == seq_status::finished) {
        return std::unexpected(seq_errc::already_finished);
    }

    if (status_ != seq_status::waiting) {
        return std::unexpected(seq_errc::invalid_state_transition);
    }

    if (scheduled_token_count_ > 0) {
        return std::unexpected(seq_errc::work_already_scheduled);
    }

    if (token_count() != cached_token_count_) {
        return std::unexpected(seq_errc::invalid_state_transition);
    }

    status_ = seq_status::running;
    assert_invariants();
    return { };
}

std::expected<void, seq_errc> seq::mark_waiting()
{
    if (status_ == seq_status::finished) {
        return std::unexpected(seq_errc::already_finished);
    }

    if (status_ != seq_status::running) {
        return std::unexpected(seq_errc::invalid_state_transition);
    }

    if (scheduled_token_count_ > 0) {
        return std::unexpected(seq_errc::work_already_scheduled);
    }

    status_ = seq_status::waiting;
    assert_invariants();
    return { };
}

std::expected<void, seq_errc>
seq::finish(finish_reason reason)
{
    if (status_ == seq_status::finished) {
        return std::unexpected(seq_errc::already_finished);
    }

    if (reason == finish_reason::none) {
        return std::unexpected(seq_errc::invalid_finish_reason);
    }

    if (scheduled_token_count_ > 0) {
        return std::unexpected(seq_errc::work_already_scheduled);
    }

    status_ = seq_status::finished;
    finish_reason_ = reason;

    assert_invariants();
    return { };
}

std::size_t seq::cached_token_count() const noexcept
{
    return cached_token_count_;
}

std::size_t seq::scheduled_token_count() const noexcept
{
    return scheduled_token_count_;
}

std::size_t seq::uncached_token_count() const noexcept
{
    return token_count() - cached_token_count_;
}

std::size_t seq::schedulable_token_count() const noexcept
{
    return token_count() - cached_token_count_ - scheduled_token_count_;
}

std::expected<void, seq_errc>
seq::schedule_tokens(std::size_t count)
{
    if (status_ == seq_status::finished) {
        return std::unexpected(seq_errc::already_finished);
    }

    if (scheduled_token_count_ != 0) {
        return std::unexpected(seq_errc::work_already_scheduled);
    }

    if (count == 0) {
        return std::unexpected(seq_errc::zero_scheduled_tokens);
    }

    if (count > schedulable_token_count()) {
        return std::unexpected(seq_errc::too_many_scheduled_tokens);
    }

    scheduled_token_count_ = count;
    assert_invariants();
    return { };
}

std::expected<void, seq_errc>
seq::commit_scheduled_tokens()
{
    if (status_ == seq_status::finished) {
        return std::unexpected(seq_errc::already_finished);
    }

    if (scheduled_token_count_ == 0) {
        return std::unexpected(seq_errc::no_work_scheduled);
    }

    cached_token_count_ += scheduled_token_count_;
    scheduled_token_count_ = 0;

    assert_invariants();
    return { };
}

void seq::cancel_scheduled_tokens() noexcept
{
    scheduled_token_count_ = 0;
    assert_invariants();
}

std::expected<void, seq_errc>
seq::set_cached_token_count(std::size_t count)
{
    if (status_ == seq_status::finished) {
        return std::unexpected(seq_errc::already_finished);
    }

    if (status_ != seq_status::waiting) {
        return std::unexpected(seq_errc::invalid_state_transition);
    }

    if (count > token_count()) {
        return std::unexpected(seq_errc::cached_token_count_out_of_range);
    }

    if (scheduled_token_count_ > 0) {
        return std::unexpected(seq_errc::work_already_scheduled);
    }

    cached_token_count_ = count;
    assert_invariants();
    return { };
}

std::size_t seq::block_size() const noexcept
{
    return block_size_;
}

std::size_t seq::logical_block_count() const noexcept
{
    return 1 + (token_count() - 1) / block_size_;
}

std::size_t seq::tokens_in_last_block() const noexcept
{
    const auto re = token_count() % block_size_;
    return (re == 0) ? block_size_ : re;
}

std::expected<std::span<const token_id>, seq_errc> seq::logical_block_tokens(std::size_t logical_block) const noexcept
{
    if (logical_block >= logical_block_count()) {
        return std::unexpected(seq_errc::index_out_of_range);
    }

    const auto start = logical_block * block_size_;
    auto end = start + block_size_;
    if (end > token_count()) {
        end = token_count();
    }

    return std::span(tokens_).subspan(start, end - start);
}

std::span<const block_id> seq::block_table() const noexcept
{
    return {
        block_table_.data(),
        block_table_.size(),
    };
}

std::expected<void, seq_errc>
seq::append_physical_block(block_id physical_block)
{
    if (status_ == seq_status::finished) {
        return std::unexpected(seq_errc::already_finished);
    }

    if (block_table_.size() >= logical_block_count()) {
        return std::unexpected(seq_errc::block_table_full);
    }

    block_table_.push_back(physical_block);
    assert_invariants();
    return { };
}

std::expected<void, seq_errc>
seq::reset_cache_metadata()
{
    if (status_ == seq_status::running) {
        return std::unexpected(seq_errc::invalid_state_transition);
    }

    if (scheduled_token_count_ != 0) {
        return std::unexpected(seq_errc::work_already_scheduled);
    }

    cached_token_count_ = 0;
    block_table_.clear();

    assert_invariants();
    return { };
}

const sampling_params& seq::params() const noexcept
{
    return params_;
}

finish_reason seq::evaluate_stop(token_id eos_token) const noexcept
{
    if (completion_token_count() == 0) {
        return finish_reason::none;
    }

    if (!params_.ignore_eos && last_token() == eos_token) {
        return finish_reason::eos;
    }

    if (completion_token_count() >= params_.max_new_tokens) {
        return finish_reason::len_limit;
    }

    return finish_reason::none;
}

void seq::assert_invariants() const noexcept
{
    assert(!tokens_.empty());
    assert(prompt_token_count_ > 0);
    assert(prompt_token_count_ <= tokens_.size());

    assert(block_size_ > 0);
    assert(cached_token_count_ <= tokens_.size());
    assert(cached_token_count_ + scheduled_token_count_ <= tokens_.size());

    assert(block_table_.size() <= logical_block_count());

    if (status_ == seq_status::finished) {
        assert(finish_reason_ != finish_reason::none);
        assert(scheduled_token_count_ == 0);
    } else {
        assert(finish_reason_ == finish_reason::none);
    }
}

} // namespace chibillm
