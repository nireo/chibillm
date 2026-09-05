#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
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
    std::optional<std::size_t> logits_index;

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
    [[nodiscard]] std::size_t sample_count() const noexcept;
};

enum class model_batch_errc : std::uint8_t {
    empty_batch,
    inconsistent_batch,
    zero_token_count,
    duplicate_sequence_id,
    unknown_sequence,
    invalid_sequence_state,
    scheduled_token_count_mismatch,
    token_range_out_of_bounds,
    position_out_of_range,
};

struct paged_batch_metadata {
    std::vector<std::uint32_t> slots;
    std::vector<std::uint32_t> block_table;
    std::vector<std::uint32_t> table_offsets;
    std::vector<std::uint32_t> table_lengths;
    std::vector<std::size_t> logits_indices;
};

result<paged_batch_metadata, model_batch_errc> prepare_paged_batch(const model_batch& batch,
                                                                   std::size_t max_positions,
                                                                   std::size_t block_count,
                                                                   std::size_t block_size);

// snapshots scheduler-owned work into backend-owned contiguous arrays.
[[nodiscard]] result<model_batch, model_batch_errc>
build_model_batch(const scheduled_batch& scheduled, const scheduler& engine);

} // namespace chibillm
