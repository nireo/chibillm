#pragma once

#include "block_manager.h"
#include "metal/metal_kv_cache.h"
#include "qwen/qwen_configs.h"

#include <optional>
#include <unordered_map>

namespace chibillm {

struct qwen3_5_linear_state {
    // f32 [2 * key_heads * key_dim + value_heads * value_dim, conv_kernel]
    metal_tensor convolution;
    // f32 [value_heads, key_dim, value_dim]
    metal_tensor recurrent;
};

// The context must outlive this state. All operations are serialized by the
// scheduler. GPU work must have finished (including failed compute passes)
// before begin/commit/abort/release accesses CPU-visible storage.
class qwen3_5_model_state final : public model_state {
public:
    [[nodiscard]] static result<std::unique_ptr<qwen3_5_model_state>, state_errc>
    make(const metal_context& context,
         const qwen3_5_config& config,
         std::size_t block_count,
         std::size_t block_size);

    result<void, state_errc> reserve(seq_id id, std::size_t tokens) override;
    // Release requires no active batch. Unknown IDs are harmless.
    void release(seq_id id) noexcept override;
    std::size_t block_size() const noexcept override;
    sequence_resources resources(seq_id id) const noexcept override;

    result<void, state_errc> begin_batch(const model_batch& batch) override;
    void commit_batch() noexcept override;
    void abort_batch() noexcept override;

    // Accept original model layer indices, not compact cache indices.
    [[nodiscard]] qwen3_5_linear_state* linear_state(seq_id id, std::size_t layer) noexcept;
    [[nodiscard]] const qwen3_5_linear_state* linear_state(seq_id id,
                                                           std::size_t layer) const noexcept;
    [[nodiscard]] std::optional<std::size_t> cache_layer(std::size_t layer) const noexcept;
    [[nodiscard]] std::optional<std::size_t> committed_tokens(seq_id id) const noexcept;
    [[nodiscard]] std::size_t sequence_count() const noexcept;
    [[nodiscard]] std::size_t linear_layer_count() const noexcept;
    [[nodiscard]] metal_kv_cache& cache() noexcept;
    [[nodiscard]] const metal_kv_cache& cache() const noexcept;

private:
    struct sequence_state {
        std::vector<qwen3_5_linear_state> layers;
        std::size_t reserved_tokens = 0;
        std::size_t committed_tokens = 0;
    };

    struct layer_snapshot {
        std::vector<std::byte> convolution;
        std::vector<std::byte> recurrent;
    };

    struct sequence_snapshot {
        seq_id id;
        std::size_t end_position;
        std::vector<layer_snapshot> layers;
    };

    qwen3_5_model_state(const metal_context& context,
                        qwen3_5_config config,
                        block_manager pages,
                        metal_kv_cache cache,
                        tensor_descriptor convolution,
                        tensor_descriptor recurrent);

    const metal_context& context_;
    qwen3_5_config config_;
    block_manager pages_;
    metal_kv_cache cache_;
    tensor_descriptor convolution_descriptor_;
    tensor_descriptor recurrent_descriptor_;
    std::vector<std::optional<std::size_t>> linear_indices_;
    std::vector<std::optional<std::size_t>> cache_indices_;
    std::size_t linear_count_ = 0;
    std::unordered_map<seq_id, sequence_state> sequences_;
    std::vector<sequence_snapshot> snapshots_;
    bool batch_open_ = false;
};
} // namespace chibillm
