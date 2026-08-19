#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <span>
#include <vector>

#include "seq.h"

namespace chibillm {

// metadata for one physical block in the global KV cache.
struct kv_block {
    block_id id;
    std::size_t ref_count { 0 };

    [[nodiscard]] bool is_free() const noexcept;
};

enum class block_manager_errc : std::uint8_t {
    invalid_block_count,
    invalid_block_size,
    too_many_blocks,

    incompatible_block_size,
    inconsistent_block_table,

    insufficient_free_blocks,
    invalid_block_id,
    block_not_in_use,

    sequence_finished,
    sequence_running,
    sequence_has_scheduled_work,
    sequence_update_failed,
};

// block_manager manages KV cache blocks
struct block_manager {
    [[nodiscard]]
    static std::expected<block_manager, block_manager_errc> make(
        std::size_t block_count,
        std::size_t block_size);

    block_manager(const block_manager&) = delete;
    block_manager& operator=(const block_manager&) = delete;

    block_manager(block_manager&&) = default;
    block_manager& operator=(block_manager&&) = default;

    [[nodiscard]] std::size_t block_count() const noexcept;
    [[nodiscard]] std::size_t block_size() const noexcept;
    [[nodiscard]] std::size_t free_block_count() const noexcept;
    [[nodiscard]] std::size_t used_block_count() const noexcept;

    [[nodiscard]] std::span<const kv_block> blocks() const noexcept;

    [[nodiscard]]
    std::expected<std::size_t, block_manager_errc>
    additional_blocks_required(const seq& sequence) const noexcept;

    [[nodiscard]]
    std::expected<bool, block_manager_errc>
    can_ensure_capacity(const seq& sequence) const noexcept;

    [[nodiscard]]
    std::expected<void, block_manager_errc>
    ensure_capacity(seq& sequence);

    [[nodiscard]]
    std::expected<void, block_manager_errc>
    release(seq& sequence);

private:
    block_manager(
        std::size_t block_count,
        std::size_t block_size);

    void assert_invariants() const noexcept;
    std::size_t block_size_;
    std::vector<kv_block> blocks_;
    std::deque<block_id> free_block_ids_;
};

} // namespace chibillm
