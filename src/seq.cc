#include "seq.h"

#include <cassert>
#include <cmath>
#include <utility>

namespace chibillm {
seq::seq(seq_id id, std::vector<token_id> prompt_tokens, generation_params params)
    : id_(id)
    , tokens_(std::move(prompt_tokens))
    , prompt_token_count_(tokens_.size())
    , params_(params)
{
    assert_invariants();
}

result<seq, seq_errc>
seq::make(seq_id id, std::vector<token_id> prompt_tokens, generation_params params)
{
    if (prompt_tokens.empty()) {
        return fail(seq_errc::empty_prompt);
    }

    if (params.max_new_tokens == 0) {
        return fail(seq_errc::invalid_max_new_tokens);
    }

    return seq {
        id,
        std::move(prompt_tokens),
        params,
    };
}

seq_id
seq::id() const noexcept
{
    return id_;
}

seq_status
seq::status() const noexcept
{
    return status_;
}

finish_reason
seq::reason() const noexcept
{
    return finish_reason_;
}

bool
seq::is_finished() const noexcept
{
    return status_ == seq_status::finished;
}

std::span<const token_id>
seq::tokens() const noexcept
{
    return {
        tokens_.data(),
        tokens_.size(),
    };
}

std::span<const token_id>
seq::prompt_tokens() const noexcept
{
    return {
        tokens_.data(),
        prompt_token_count_,
    };
}

std::span<const token_id>
seq::completion_tokens() const noexcept
{
    return {
        tokens_.data() + prompt_token_count_,
        tokens_.size() - prompt_token_count_,
    };
}

token_id
seq::last_token() const noexcept
{
    return tokens_.back();
}

std::size_t
seq::token_count() const noexcept
{
    return tokens_.size();
}

std::size_t
seq::prompt_token_count() const noexcept
{
    return prompt_token_count_;
}

std::size_t
seq::completion_token_count() const noexcept
{
    return tokens_.size() - prompt_token_count_;
}

result<void, seq_errc>
seq::append_token(token_id token)
{
    if (status_ == seq_status::finished) {
        return fail(seq_errc::already_finished);
    }

    if (status_ != seq_status::running) {
        return fail(seq_errc::invalid_state_transition);
    }

    if (scheduled_token_count_ != 0) {
        return fail(seq_errc::work_already_scheduled);
    }

    if (processed_token_count_ != token_count()) {
        return fail(seq_errc::invalid_state_transition);
    }

    tokens_.push_back(token);
    assert_invariants();
    return {};
}

result<void, seq_errc>
seq::mark_running()
{
    if (status_ == seq_status::finished) {
        return fail(seq_errc::already_finished);
    }

    if (status_ != seq_status::waiting) {
        return fail(seq_errc::invalid_state_transition);
    }

    if (scheduled_token_count_ > 0) {
        return fail(seq_errc::work_already_scheduled);
    }

    if (token_count() != processed_token_count_) {
        return fail(seq_errc::invalid_state_transition);
    }

    status_ = seq_status::running;
    assert_invariants();
    return {};
}

result<void, seq_errc>
seq::finish(finish_reason reason)
{
    if (status_ == seq_status::finished) {
        return fail(seq_errc::already_finished);
    }

    if (reason == finish_reason::none) {
        return fail(seq_errc::invalid_finish_reason);
    }

    if (scheduled_token_count_ > 0) {
        return fail(seq_errc::work_already_scheduled);
    }

    status_ = seq_status::finished;
    finish_reason_ = reason;

    assert_invariants();
    return {};
}

std::size_t
seq::processed_token_count() const noexcept
{
    return processed_token_count_;
}

std::size_t
seq::scheduled_token_count() const noexcept
{
    return scheduled_token_count_;
}

std::size_t
seq::unprocessed_token_count() const noexcept
{
    return token_count() - processed_token_count_;
}

std::size_t
seq::schedulable_token_count() const noexcept
{
    return token_count() - processed_token_count_ - scheduled_token_count_;
}

result<void, seq_errc>
seq::schedule_tokens(std::size_t count)
{
    if (status_ == seq_status::finished) {
        return fail(seq_errc::already_finished);
    }

    if (scheduled_token_count_ != 0) {
        return fail(seq_errc::work_already_scheduled);
    }

    if (count == 0) {
        return fail(seq_errc::zero_scheduled_tokens);
    }

    if (count > schedulable_token_count()) {
        return fail(seq_errc::too_many_scheduled_tokens);
    }

    scheduled_token_count_ = count;
    assert_invariants();
    return {};
}

result<void, seq_errc>
seq::commit_scheduled_tokens()
{
    if (status_ == seq_status::finished) {
        return fail(seq_errc::already_finished);
    }

    if (scheduled_token_count_ == 0) {
        return fail(seq_errc::no_work_scheduled);
    }

    processed_token_count_ += scheduled_token_count_;
    scheduled_token_count_ = 0;

    assert_invariants();
    return {};
}

void
seq::cancel_scheduled_tokens() noexcept
{
    scheduled_token_count_ = 0;
    assert_invariants();
}

const generation_params&
seq::params() const noexcept
{
    return params_;
}

finish_reason
seq::evaluate_stop(token_id eos_token) const noexcept
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

void
seq::assert_invariants() const noexcept
{
#ifndef NDEBUG
    assert(!tokens_.empty());
    assert(prompt_token_count_ > 0);
    assert(prompt_token_count_ <= tokens_.size());

    assert(processed_token_count_ <= tokens_.size());
    assert(processed_token_count_ + scheduled_token_count_ <= tokens_.size());

    if (status_ == seq_status::finished) {
        assert(finish_reason_ != finish_reason::none);
        assert(scheduled_token_count_ == 0);
    } else {
        assert(finish_reason_ == finish_reason::none);
    }
#endif
}

} // namespace chibillm
