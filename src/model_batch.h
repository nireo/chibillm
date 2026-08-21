#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "result.h"
#include "scheduler.h"

namespace chibillm {

using position_id = std::uint32_t;

struct model_batch_item {
    seq_id id;

    // range in model_batch::tokens and model_batch::positions.
    std::size_t token_offset;
    std::size_t token_count;

    // flattened token whose output logits are sampled for this sequence.
    std::size_t logits_index;

    // copied so backend work does not depend on scheduler container lifetime.
    std::vector<block_id> block_table;
};

// owns all metadata needed by one backend invocation.
struct model_batch {
    batch_id id;
    batch_phase phase;
    std::size_t kv_block_size;

    std::vector<token_id> tokens;
    std::vector<position_id> positions;
    std::vector<model_batch_item> items;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t sequence_count() const noexcept;
    [[nodiscard]] std::size_t token_count() const noexcept;
};

enum class model_batch_errc : std::uint8_t {
    empty_batch,
    zero_token_count,
    duplicate_sequence_id,
    unknown_sequence,
    invalid_sequence_state,
    scheduled_token_count_mismatch,
    token_range_out_of_bounds,
    position_out_of_range,
    incompatible_block_size,
    incomplete_block_table,
};

// snapshots scheduler-owned work into backend-owned contiguous arrays.
[[nodiscard]] result<model_batch, model_batch_errc>
build_model_batch(const scheduled_batch& scheduled, const scheduler& engine);

} // namespace chibillm
