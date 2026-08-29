#include "block_manager.h"

#include <cassert>
#include <limits>
#include <vector>

namespace chibillm {

bool
kv_block::is_free() const noexcept
{
    return ref_count == 0;
}

result<block_manager, block_manager_errc>
block_manager::make(std::size_t block_count, std::size_t block_size)
{
    if (block_count == 0) {
        return fail(block_manager_errc::invalid_block_count);
    }

    if (block_size == 0) {
        return fail(block_manager_errc::invalid_block_size);
    }

    // validate the largest id before narrowing indices to block_id.
    if (block_count - 1 > std::numeric_limits<block_id>::max()) {
        return fail(block_manager_errc::too_many_blocks);
    }

    return block_manager {
        block_count,
        block_size,
    };
}

block_manager::block_manager(std::size_t block_count, std::size_t block_size)
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

std::size_t
block_manager::block_count() const noexcept
{
    return blocks_.size();
}

std::size_t
block_manager::block_size() const noexcept
{
    return block_size_;
}

std::size_t
block_manager::free_block_count() const noexcept
{
    return free_block_ids_.size();
}

std::size_t
block_manager::used_block_count() const noexcept
{
    return blocks_.size() - free_block_ids_.size();
}

std::span<const kv_block>
block_manager::blocks() const noexcept
{
    return {
        blocks_.data(),
        blocks_.size(),
    };
}

result<std::size_t, block_manager_errc>
block_manager::additional_blocks_required(const seq& sequence) const noexcept
{
    if (sequence.block_size() != block_size_) {
        return fail(block_manager_errc::incompatible_block_size);
    }

    const auto table = sequence.block_table();
    const auto log_count = sequence.logical_block_count();

    if (table.size() > log_count) {
        return fail(block_manager_errc::inconsistent_block_table);
    }

    for (const auto physical_id : table) {
        if (physical_id >= blocks_.size()) {
            return fail(block_manager_errc::invalid_block_id);
        }

        if (blocks_[physical_id].is_free()) {
            return fail(block_manager_errc::block_not_in_use);
        }
    }

    return log_count - table.size();
}

result<bool, block_manager_errc>
block_manager::can_ensure_capacity(const seq& sequence) const noexcept
{
    const auto amount_needed = additional_blocks_required(sequence);
    if (!amount_needed) {
        return fail(amount_needed.error());
    }

    return *amount_needed <= free_block_ids_.size();
}

result<void, block_manager_errc>
block_manager::ensure_capacity(seq& sequence)
{
    if (sequence.status() == seq_status::finished) {
        return fail(block_manager_errc::sequence_finished);
    }

    const auto amount_needed = additional_blocks_required(sequence);
    if (!amount_needed) {
        return fail(amount_needed.error());
    }

    const auto amount = *amount_needed;
    if (amount == 0) {
        return {};
    }

    if (amount > free_block_ids_.size()) {
        return fail(block_manager_errc::insufficient_free_blocks);
    }

    for (std::size_t i = 0; i < amount; ++i) {
        const auto free_id = free_block_ids_.front();
        auto& fblock = blocks_[free_id];
        assert(fblock.is_free());

        // update the sequence before taking manager ownership.
        const auto ok = sequence.append_physical_block(fblock.id);
        if (!ok) {
            assert(false && "prevalidated physical-block append failed");
            return fail(block_manager_errc::sequence_update_failed);
        }

        free_block_ids_.pop_front();
        fblock.ref_count = 1;
    }

    assert_invariants();
    return {};
}

result<void, block_manager_errc>
block_manager::release(seq& sequence)
{
    if (sequence.status() == seq_status::running) {
        return fail(block_manager_errc::sequence_running);
    }

    if (sequence.scheduled_token_count() != 0) {
        return fail(block_manager_errc::sequence_has_scheduled_work);
    }

    if (sequence.block_size() != block_size_) {
        return fail(block_manager_errc::incompatible_block_size);
    }

    // validate the full table before changing any refcount.
    std::vector<bool> seen(blocks_.size(), false);
    for (const auto id : sequence.block_table()) {
        if (id >= blocks_.size()) {
            return fail(block_manager_errc::invalid_block_id);
        }

        if (seen[id]) {
            return fail(block_manager_errc::inconsistent_block_table);
        }
        seen[id] = true;

        if (blocks_[id].is_free()) {
            return fail(block_manager_errc::block_not_in_use);
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
        return fail(block_manager_errc::sequence_update_failed);
    }

    assert_invariants();
    return {};
}

void
block_manager::assert_invariants() const noexcept
{
#ifndef NDEBUG
    assert(block_size_ > 0);

    // refcounts and free-list membership must describe the same state.
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
