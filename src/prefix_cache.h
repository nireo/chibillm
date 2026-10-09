#pragma once

#include "model_state.h"

#include <list>
#include <memory>
#include <span>
#include <vector>

namespace chibillm {

// Small, byte-bounded LRU. Exact token comparison avoids hash collisions, and
// checkpoints are copied on restore, so entries never own active GPU pages.
class prefix_cache {
public:
    explicit prefix_cache(std::size_t capacity_bytes) noexcept;
    const model_checkpoint* find(std::span<const token_id> prompt);
    bool contains(std::span<const token_id> prefix) const noexcept;
    bool fits(std::size_t token_count, std::size_t state_bytes) const noexcept;
    void make_room(std::size_t token_count, std::size_t state_bytes) noexcept;
    void insert(std::span<const token_id> prefix, std::unique_ptr<model_checkpoint> state);

    std::size_t
    size_bytes() const noexcept
    {
        return bytes_;
    }

private:
    struct entry {
        std::vector<token_id> tokens;
        std::unique_ptr<model_checkpoint> state;
        std::size_t size_bytes() const noexcept;
    };

    std::size_t capacity_;
    std::size_t bytes_ = 0;
    std::list<entry> entries_;
};

} // namespace chibillm
