#include "block_manager.h"

#include <cassert>
#include <limits>
#include <vector>

namespace chibillm {

bool kv_block::is_free() const noexcept
{
    return ref_count == 0;
}

std::expected<block_manager, block_manager_errc>
block_manager::make(
    std::size_t block_count,
    std::size_t block_size)
{
    if (block_count == 0) {
        return std::unexpected(block_manager_errc::invalid_block_count);
    }

    if (block_size == 0) {
        return std::unexpected(block_manager_errc::invalid_block_size);
    }

    // IDs range from zero through block_count - 1. Validate the largest ID before
    // narrowing each size_t index to block_id in the constructor.
    if (block_count - 1 > std::numeric_limits<block_id>::max()) {
        return std::unexpected(block_manager_errc::too_many_blocks);
    }

    return block_manager {
        block_count,
        block_size,
    };
}

block_manager::block_manager(
    std::size_t block_count,
    std::size_t block_size)
    : block_size_(block_size)
{
    blocks_.reserve(block_count);

    for (std::size_t index = 0; index < block_count; ++index) {
        const auto id = static_cast<block_id>(index);

        blocks_.push_back(kv_block {
            .id = id,
            .ref_count = 0,
        });
        free_block_ids_.push_back(id);
    }

    assert_invariants();
}

std::size_t block_manager::block_count() const noexcept
{
    return blocks_.size();
}

std::size_t block_manager::block_size() const noexcept
{
    return block_size_;
}

std::size_t block_manager::free_block_count() const noexcept
{
    return free_block_ids_.size();
}

std::size_t block_manager::used_block_count() const noexcept
{
    return blocks_.size() - free_block_ids_.size();
}

std::span<const kv_block> block_manager::blocks() const noexcept
{
    return {
        blocks_.data(),
        blocks_.size(),
    };
}

std::expected<std::size_t, block_manager_errc>
block_manager::additional_blocks_required(
    const seq& sequence) const noexcept
{
    if (sequence.block_size() != block_size_) {
        return std::unexpected(block_manager_errc::incompatible_block_size);
    }

    const auto table = sequence.block_table();
    const auto log_count = sequence.logical_block_count();

    if (table.size() > log_count) {
        return std::unexpected(block_manager_errc::inconsistent_block_table);
    }

    for (const auto physical_id : table) {
        if (physical_id >= blocks_.size()) {
            return std::unexpected(block_manager_errc::invalid_block_id);
        }

        if (blocks_[physical_id].is_free()) {
            return std::unexpected(block_manager_errc::block_not_in_use);
        }
    }

    return log_count - table.size();
}

std::expected<bool, block_manager_errc>
block_manager::can_ensure_capacity(
    const seq& sequence) const noexcept
{
    const auto amount_needed = additional_blocks_required(sequence);
    if (!amount_needed) {
        return std::unexpected(amount_needed.error());
    }

    return *amount_needed <= free_block_ids_.size();
}

// -----------------------------------------------------------------------------
// Allocation
// -----------------------------------------------------------------------------

std::expected<void, block_manager_errc>
block_manager::ensure_capacity(seq& sequence)
{
    if (sequence.status() == seq_status::finished) {
        return std::unexpected(block_manager_errc::sequence_finished);
    }

    const auto amount_needed = additional_blocks_required(sequence);
    if (!amount_needed) {
        return std::unexpected(amount_needed.error());
    }

    const auto amount = *amount_needed;
    if (amount == 0) {
        // we don't need to do anything here
        return { };
    }

    if (amount > free_block_ids_.size()) {
        return std::unexpected(block_manager_errc::insufficient_free_blocks);
    }

    for (std::size_t i = 0; i < amount; ++i) {
        const auto free_id = free_block_ids_.front();
        auto& fblock = blocks_[free_id];
        assert(fblock.is_free());

        // Append before changing manager ownership. All domain preconditions were
        // checked above, so failure indicates an internal contract mismatch.
        const auto ok = sequence.append_physical_block(fblock.id);
        if (!ok) {
            assert(false && "prevalidated physical-block append failed");
            return std::unexpected(block_manager_errc::sequence_update_failed);
        }

        free_block_ids_.pop_front();
        fblock.ref_count = 1;
    }

    assert_invariants();
    return { };
}

// -----------------------------------------------------------------------------
// Release
// -----------------------------------------------------------------------------

std::expected<void, block_manager_errc>
block_manager::release(seq& sequence)
{
    if (sequence.status() == seq_status::running) {
        return std::unexpected(block_manager_errc::sequence_running);
    }

    if (sequence.scheduled_token_count() != 0) {
        return std::unexpected(block_manager_errc::sequence_has_scheduled_work);
    }

    if (sequence.block_size() != block_size_) {
        return std::unexpected(block_manager_errc::incompatible_block_size);
    }

    // Validate the complete table before modifying any refcount so ordinary
    // errors cannot leave a partially released sequence.
    std::vector<bool> seen(blocks_.size(), false);
    for (const auto id : sequence.block_table()) {
        if (id >= blocks_.size()) {
            return std::unexpected(block_manager_errc::invalid_block_id);
        }

        if (seen[id]) {
            return std::unexpected(block_manager_errc::inconsistent_block_table);
        }
        seen[id] = true;

        if (blocks_[id].is_free()) {
            return std::unexpected(block_manager_errc::block_not_in_use);
        }
    }

    for (const auto id : sequence.block_table()) {
        auto& block = blocks_[id];
        --block.ref_count;
        if (block.ref_count == 0) {
            free_block_ids_.push_back(block.id);
        }
    }

    const auto ok = sequence.reset_cache_metadata();
    if (!ok) {
        assert(false && "prevalidated sequence cache reset failed");
        return std::unexpected(block_manager_errc::sequence_update_failed);
    }

    assert_invariants();
    return { };
}

// -----------------------------------------------------------------------------
// Internal consistency
// -----------------------------------------------------------------------------

void block_manager::assert_invariants() const noexcept
{
#ifndef NDEBUG
    assert(block_size_ > 0);

    // A zero refcount and free-list membership are two representations of the same
    // state. Validate that they agree and that the free list contains no duplicate
    // or out-of-range IDs.
    std::vector<bool> seen_free(blocks_.size(), false);
    for (const auto id : free_block_ids_) {
        assert(id < blocks_.size());
        assert(!seen_free[id]);
        seen_free[id] = true;
        assert(blocks_[id].is_free());
    }

    for (std::size_t index = 0; index < blocks_.size(); ++index) {
        const auto& block = blocks_[index];
        assert(block.id == static_cast<block_id>(index));
        assert(block.is_free() == seen_free[index]);
    }
#endif
}

} // namespace chibillm
