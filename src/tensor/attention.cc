#include "tensor/attention.h"
#include "metal/metal_kernels.h"
#include <cstring>
#include <vector>

namespace chibillm {
result<attention_metadata_tensors, tensor_op_errc>
upload_attention_metadata(const metal_context& context, attention_metadata metadata)
{
    auto positions = upload_u32(context, metadata.positions);
    if (!positions)
        return fail(positions.error());
    auto slots = upload_u32(context, metadata.slots);
    if (!slots)
        return fail(slots.error());
    auto block_table = upload_u32(context, metadata.block_table);
    if (!block_table)
        return fail(block_table.error());
    auto table_offsets = upload_u32(context, metadata.block_table_offsets);
    if (!table_offsets)
        return fail(table_offsets.error());
    auto table_lengths = upload_u32(context, metadata.block_table_lengths);
    if (!table_lengths)
        return fail(table_lengths.error());
    return attention_metadata_tensors { std::move(*positions), std::move(*slots),
                                        std::move(*block_table), std::move(*table_offsets),
                                        std::move(*table_lengths) };
}

namespace {
result<void, tensor_op_errc>
validate_shapes(const metal_tensor& queries,
                const metal_tensor& positions,
                const metal_tensor& block_table_offsets,
                const metal_tensor& block_table_lengths,
                std::size_t layer,
                std::size_t query_head_count,
                const metal_kv_cache& cache,
                const metal_tensor& output)
{
    const auto& query_shape = queries.descriptor().shape();
    const auto& position_shape = positions.descriptor().shape();
    const auto& offset_shape = block_table_offsets.descriptor().shape();
    const auto& length_shape = block_table_lengths.descriptor().shape();
    const auto& output_shape = output.descriptor().shape();

    if (query_shape.rank() != 2
        || position_shape.rank() != 1
        || offset_shape.rank() != 1
        || length_shape.rank() != 1
        || output_shape.rank() != 2) {
        return fail(tensor_op_errc::invalid_rank);
    }

    if (queries.descriptor().type() != dtype::f32
        || positions.descriptor().type() != dtype::u32
        || block_table_offsets.descriptor().type() != dtype::u32
        || block_table_lengths.descriptor().type() != dtype::u32
        || output.descriptor().type() != dtype::f32) {
        return fail(tensor_op_errc::unsupported_dtype);
    }

    const auto rows = query_shape.dimensions()[0];
    const auto query_feature_count = query_shape.dimensions()[1];
    if (position_shape.dimensions()[0] != rows) {
        return fail(tensor_op_errc::position_count_mismatch);
    }
    if (offset_shape.dimensions()[0] != rows || length_shape.dimensions()[0] != rows) {
        return fail(tensor_op_errc::block_table_metadata_count_mismatch);
    }

    if (output_shape.dimensions()[0] != rows
        || output_shape.dimensions()[1] != query_feature_count) {
        return fail(tensor_op_errc::output_shape_mismatch);
    }

    if (query_head_count == 0) {
        return fail(tensor_op_errc::invalid_head_count);
    }
    if (query_feature_count % query_head_count != 0) {
        return fail(tensor_op_errc::invalid_head_dimension);
    }

    const auto head_dimension = query_feature_count / query_head_count;
    if (head_dimension != cache.head_dimension()) {
        return fail(tensor_op_errc::cache_head_dimension_mismatch);
    }
    if (query_head_count % cache.kv_head_count() != 0) {
        return fail(tensor_op_errc::invalid_kv_head_mapping);
    }
    if (layer >= cache.layer_count()) {
        return fail(tensor_op_errc::cache_layer_out_of_range);
    }

    return {};
}
} // namespace

result<prepared_attention_batch, tensor_op_errc>
prepared_attention_batch::make(const metal_context& context,
                               const metal_tensor& positions,
                               const metal_tensor& block_table,
                               const metal_tensor& block_table_offsets,
                               const metal_tensor& block_table_lengths,
                               const metal_kv_cache& cache)
{
    for (auto* tensor : { &positions, &block_table, &block_table_offsets, &block_table_lengths }) {
        if (tensor->descriptor().shape().rank() != 1)
            return fail(tensor_op_errc::invalid_rank);
        if (tensor->descriptor().type() != dtype::u32)
            return fail(tensor_op_errc::unsupported_dtype);
    }
    const auto rows = positions.descriptor().shape().dimensions()[0];
    if (block_table_offsets.descriptor().shape().dimensions()[0] != rows
        || block_table_lengths.descriptor().shape().dimensions()[0] != rows) {
        return fail(tensor_op_errc::block_table_metadata_count_mismatch);
    }
    const auto& table_shape = block_table.descriptor().shape();
    const auto position_bytes = positions.buffer().bytes();
    const auto table_bytes = block_table.buffer().bytes();
    const auto offset_bytes = block_table_offsets.buffer().bytes();
    const auto length_bytes = block_table_lengths.buffer().bytes();
    const auto table_entry_count = table_shape.dimensions()[0];
    const auto read_u32 = [](std::span<const std::byte> bytes, std::size_t index) {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
        return value;
    };

    for (std::size_t row = 0; row < rows; ++row) {
        const auto position = read_u32(position_bytes, row);
        const auto table_offset = read_u32(offset_bytes, row);
        const auto table_length = read_u32(length_bytes, row);
        const auto required_blocks = static_cast<std::size_t>(position) / cache.block_size() + 1;

        if (table_offset > table_entry_count
            || table_length > table_entry_count - table_offset
            || required_blocks > table_length) {
            return fail(tensor_op_errc::block_table_range_out_of_bounds);
        }
    }
    for (std::size_t index = 0; index < table_entry_count; ++index) {
        if (read_u32(table_bytes, index) >= cache.block_count()) {
            return fail(tensor_op_errc::cache_block_out_of_range);
        }
    }

    // Consecutive query rows from the same sequence share a block table and
    // can reuse each paged K/V tile. Ragged batches are split at sequence or
    // position discontinuities so a tile never crosses logical sequences.
    constexpr std::size_t flash_query_tile_size = 8;
    std::vector<std::uint32_t> query_tile_starts;
    std::vector<std::uint32_t> query_tile_lengths;
    query_tile_starts.reserve((rows + flash_query_tile_size - 1) / flash_query_tile_size);
    query_tile_lengths.reserve(query_tile_starts.capacity());
    bool has_shared_query_tile = false;
    for (std::size_t row = 0; row < rows;) {
        std::size_t tile_rows = 1;
        const auto first_table_offset = read_u32(offset_bytes, row);
        const auto first_table_length = read_u32(length_bytes, row);
        while (tile_rows < flash_query_tile_size && row + tile_rows < rows) {
            const auto candidate = row + tile_rows;
            const auto previous_position = read_u32(position_bytes, candidate - 1);
            const auto candidate_position = read_u32(position_bytes, candidate);
            if (read_u32(offset_bytes, candidate) != first_table_offset
                || read_u32(length_bytes, candidate) != first_table_length
                || candidate_position != previous_position + 1) {
                break;
            }
            ++tile_rows;
        }
        query_tile_starts.push_back(static_cast<std::uint32_t>(row));
        query_tile_lengths.push_back(static_cast<std::uint32_t>(tile_rows));
        has_shared_query_tile = has_shared_query_tile || tile_rows > 1;
        row += tile_rows;
    }

    prepared_attention_batch result(positions, block_table, block_table_offsets,
                                    block_table_lengths, cache.config());
    if (has_shared_query_tile && cache.head_dimension() <= 128) {
        const auto tile_bytes = query_tile_starts.size() * sizeof(std::uint32_t);
        auto starts = context.make_shared_buffer(tile_bytes);
        auto lengths = context.make_shared_buffer(tile_bytes);
        if (!starts || !lengths)
            return fail(tensor_op_errc::backend_failure);
        std::memcpy(starts->bytes().data(), query_tile_starts.data(), tile_bytes);
        std::memcpy(lengths->bytes().data(), query_tile_lengths.data(), tile_bytes);
        result.tile_starts_ = std::move(*starts);
        result.tile_lengths_ = std::move(*lengths);
        result.tile_count_ = query_tile_starts.size();
    }
    return result;
}

result<void, tensor_op_errc>
paged_attention(const metal_context& context,
                const metal_tensor& queries,
                const prepared_attention_batch& metadata,
                std::size_t layer,
                std::size_t query_head_count,
                const metal_kv_cache& cache,
                metal_tensor& output)
{
    const auto& positions = *metadata.positions_;
    const auto& block_table = *metadata.table_;
    const auto& block_table_offsets = *metadata.offsets_;
    const auto& block_table_lengths = *metadata.lengths_;
    CL_TRY(validate_shapes(queries, positions, block_table_offsets, block_table_lengths, layer,
                           query_head_count, cache, output));
    if (metadata.geometry_.block_count != cache.block_count()
        || metadata.geometry_.block_size != cache.block_size()) {
        return fail(tensor_op_errc::cache_block_out_of_range);
    }
    const auto rows = positions.descriptor().shape().dimensions()[0];
    const auto head_dimension = cache.head_dimension();
    const auto slot_count = cache.block_count() * cache.block_size();
    const auto table_entry_count = block_table.descriptor().shape().dimensions()[0];
    if (metadata.tile_count_ && head_dimension <= 128) {
        const auto dispatched = metal_kernels(context).dispatch_paged_flash_attention_prefill_f32(
            queries.buffer(), positions.buffer(), block_table.buffer(),
            block_table_offsets.buffer(), block_table_lengths.buffer(), cache.keys().buffer(),
            cache.values().buffer(), *metadata.tile_starts_, *metadata.tile_lengths_,
            output.buffer(), rows, query_head_count, cache.kv_head_count(), head_dimension,
            cache.block_size(), slot_count, layer, table_entry_count, metadata.tile_count_);
        if (!dispatched) {
            return fail(tensor_op_errc::backend_failure);
        }
    } else {
        const auto dispatched = metal_kernels(context).dispatch_paged_attention_f32(
            queries.buffer(), positions.buffer(), block_table.buffer(),
            block_table_offsets.buffer(), block_table_lengths.buffer(), cache.keys().buffer(),
            cache.values().buffer(), output.buffer(), rows, query_head_count, cache.kv_head_count(),
            head_dimension, cache.block_size(), slot_count, layer, table_entry_count);
        if (!dispatched) {
            return fail(tensor_op_errc::backend_failure);
        }
    }

    return {};
}

result<void, tensor_op_errc>
paged_attention(const metal_context& context,
                const metal_tensor& queries,
                const metal_tensor& positions,
                const metal_tensor& block_table,
                const metal_tensor& block_table_offsets,
                const metal_tensor& block_table_lengths,
                std::size_t layer,
                std::size_t query_head_count,
                const metal_kv_cache& cache,
                metal_tensor& output)
{
    CL_TRY(validate_shapes(queries, positions, block_table_offsets, block_table_lengths, layer,
                           query_head_count, cache, output));
    auto metadata = prepared_attention_batch::make(context, positions, block_table,
                                                   block_table_offsets, block_table_lengths, cache);
    if (!metadata)
        return fail(metadata.error());
    return paged_attention(context, queries, *metadata, layer, query_head_count, cache, output);
}
} // namespace chibillm
