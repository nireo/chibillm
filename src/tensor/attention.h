#pragma once
#include "tensor/tensor_ops.h"
#include <optional>

namespace chibillm {
struct attention_metadata {
    std::span<const std::uint32_t> positions;
    std::span<const std::uint32_t> slots;
    std::span<const std::uint32_t> block_table;
    std::span<const std::uint32_t> block_table_offsets;
    std::span<const std::uint32_t> block_table_lengths;
};

struct attention_metadata_tensors {
    metal_tensor positions;
    metal_tensor slots;
    metal_tensor block_table;
    metal_tensor table_offsets;
    metal_tensor table_lengths;
};

result<attention_metadata_tensors, tensor_op_errc>
upload_attention_metadata(const metal_context& context, attention_metadata metadata);

// Metadata tensors must remain alive and unchanged while the prepared batch is used.
class prepared_attention_batch {
public:
    static result<prepared_attention_batch, tensor_op_errc> make(const metal_context& context,
                                                                 const metal_tensor& positions,
                                                                 const metal_tensor& block_table,
                                                                 const metal_tensor& offsets,
                                                                 const metal_tensor& lengths,
                                                                 const metal_kv_cache& cache);

    const metal_tensor&
    positions() const
    {
        return *positions_;
    }

private:
    friend result<void, tensor_op_errc> paged_attention(const metal_context&,
                                                        const metal_tensor&,
                                                        const prepared_attention_batch&,
                                                        std::size_t,
                                                        std::size_t,
                                                        const metal_kv_cache&,
                                                        metal_tensor&);

    prepared_attention_batch(const metal_tensor& positions,
                             const metal_tensor& table,
                             const metal_tensor& offsets,
                             const metal_tensor& lengths,
                             kv_cache_config geometry)
        : positions_(&positions)
        , table_(&table)
        , offsets_(&offsets)
        , lengths_(&lengths)
        , geometry_(geometry)
    {}

    const metal_tensor* positions_;
    const metal_tensor* table_;
    const metal_tensor* offsets_;
    const metal_tensor* lengths_;
    kv_cache_config geometry_;
    std::optional<metal_buffer> tile_starts_;
    std::optional<metal_buffer> tile_lengths_;
    std::size_t tile_count_ = 0;
};

result<void, tensor_op_errc> paged_attention(const metal_context& context,
                                             const metal_tensor& queries,
                                             const prepared_attention_batch& metadata,
                                             std::size_t layer,
                                             std::size_t query_head_count,
                                             const metal_kv_cache& cache,
                                             metal_tensor& output);
} // namespace chibillm
