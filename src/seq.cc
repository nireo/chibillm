#include "seq.h"

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

    if (params.temperature <= 0.0f) {
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

size_t seq::logical_block_count() const noexcept
{
    return (token_count() + block_size_ - 1) / block_size_;
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
    // Construction rejects empty prompts and tokens are never removed.
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

std::size_t seq::cached_token_count() const noexcept
{
    return cached_token_count_;
}

std::size_t seq::scheduled_token_count() const noexcept
{
    return scheduled_token_count_;
}

std::size_t seq::block_size() const noexcept
{
    return block_size_;
}

std::span<const block_id> seq::block_table() const noexcept
{
    return {
        block_table_.data(),
        block_table_.size(),
    };
}

const sampling_params& seq::params() const noexcept
{
    return params_;
}
}
