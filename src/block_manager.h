#pragma once
#include "model_state.h"
#include <deque>
#include <unordered_map>
#include <vector>

namespace chibillm {
enum class block_manager_errc {
    invalid_block_count,
    invalid_block_size,
    too_many_blocks
};

class block_manager final : public model_state {
public:
    static result<block_manager, block_manager_errc> make(std::size_t block_count,
                                                          std::size_t block_size);
    block_manager(block_manager&&) noexcept = default;
    block_manager& operator=(block_manager&&) noexcept = default;

    std::size_t
    block_count() const noexcept
    {
        return block_count_;
    }

    std::size_t
    block_size() const noexcept override
    {
        return block_size_;
    }

    std::size_t
    free_block_count() const noexcept
    {
        return free_.size();
    }

    std::size_t
    used_block_count() const noexcept
    {
        return block_count_ - free_.size();
    }

    sequence_resources resources(seq_id id) const noexcept override;
    result<void, state_errc> reserve(seq_id id, std::size_t token_count) override;
    void release(seq_id id) noexcept override;

private:
    block_manager(std::size_t count, std::size_t size);
    std::size_t block_count_;
    std::size_t block_size_;
    std::deque<block_id> free_;
    std::unordered_map<seq_id, std::vector<block_id>> tables_;
};
} // namespace chibillm
