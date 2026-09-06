#include "block_manager.h"
#include <limits>

namespace chibillm {
result<block_manager, block_manager_errc>
block_manager::make(std::size_t count, std::size_t size)
{
    if (!count)
        return fail(block_manager_errc::invalid_block_count);
    if (!size)
        return fail(block_manager_errc::invalid_block_size);
    if (count - 1 > std::numeric_limits<block_id>::max())
        return fail(block_manager_errc::too_many_blocks);
    return block_manager(count, size);
}

block_manager::block_manager(std::size_t count, std::size_t size)
    : block_count_(count)
    , block_size_(size)
{
    for (std::size_t i = 0; i < count; ++i)
        free_.push_back(static_cast<block_id>(i));
}

sequence_resources
block_manager::resources(seq_id id) const noexcept
{
    auto found = tables_.find(id);
    return found == tables_.end() ? sequence_resources { } : sequence_resources { found->second };
}

result<void, state_errc>
block_manager::reserve(seq_id id, std::size_t tokens)
{
    if (!tokens)
        return fail(state_errc::invalid_reservation);
    const auto required = 1 + (tokens - 1) / block_size_;
    const auto existing = resources(id).blocks.size();
    if (required < existing)
        return fail(state_errc::invalid_reservation);
    const auto additional = required - existing;
    if (additional > free_.size())
        return fail(state_errc::capacity_exhausted);
    if (!additional)
        return { };
    auto& table = tables_[id];
    table.reserve(required);
    for (std::size_t i = 0; i < additional; ++i) {
        table.push_back(free_.front());
        free_.pop_front();
    }
    return { };
}

void block_manager::release(seq_id id) noexcept
{
    const auto found = tables_.find(id);
    if (found == tables_.end())
        return;
    for (auto block : found->second)
        free_.push_back(block);
    tables_.erase(found);
}
} // namespace chibillm
