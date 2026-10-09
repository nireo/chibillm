#include "prefix_cache.h"

#include <algorithm>

namespace chibillm {

prefix_cache::prefix_cache(std::size_t capacity_bytes) noexcept
    : capacity_(capacity_bytes)
{}

std::size_t
prefix_cache::entry::size_bytes() const noexcept
{
    return tokens.size() * sizeof(token_id) + state->size_bytes();
}

const model_checkpoint*
prefix_cache::find(std::span<const token_id> prompt)
{
    auto best = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        const auto count = it->tokens.size();
        // Leave a suffix to execute: state alone does not contain output logits.
        if (count < prompt.size()
            && (best == entries_.end() || count > best->tokens.size())
            && std::ranges::equal(it->tokens, prompt.first(count)))
            best = it;
    }
    if (best == entries_.end())
        return nullptr;
    entries_.splice(entries_.begin(), entries_, best);
    return entries_.front().state.get();
}

bool
prefix_cache::contains(std::span<const token_id> prefix) const noexcept
{
    return std::ranges::any_of(
        entries_, [&](const auto& item) { return std::ranges::equal(item.tokens, prefix); });
}

bool
prefix_cache::fits(std::size_t tokens, std::size_t state_bytes) const noexcept
{
    return state_bytes
        && state_bytes <= capacity_
        && tokens <= (capacity_ - state_bytes) / sizeof(token_id);
}

void
prefix_cache::make_room(std::size_t tokens, std::size_t state_bytes) noexcept
{
    if (!fits(tokens, state_bytes))
        return;
    const auto needed = tokens * sizeof(token_id) + state_bytes;
    while (bytes_ > capacity_ - needed) {
        bytes_ -= entries_.back().size_bytes();
        entries_.pop_back();
    }
}

void
prefix_cache::insert(std::span<const token_id> prefix, std::unique_ptr<model_checkpoint> state)
{
    if (!state
        || state->token_count != prefix.size()
        || !fits(prefix.size(), state->size_bytes())
        || contains(prefix))
        return;
    make_room(prefix.size(), state->size_bytes());
    entries_.push_front({ { prefix.begin(), prefix.end() }, std::move(state) });
    bytes_ += entries_.front().size_bytes();
}

} // namespace chibillm
