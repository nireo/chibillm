#include "metal/metal_dispatch_internal.h"
#include "metal/metal_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace chibillm {
using namespace metal_dispatch;

result<void, metal_error>
metal_kernels::dispatch_store_kv_f32(const metal_buffer& keys,
                                     const metal_buffer& values,
                                     const metal_buffer& slot_mapping,
                                     metal_buffer& key_cache,
                                     metal_buffer& value_cache,
                                     std::size_t rows,
                                     std::size_t feature_count,
                                     std::size_t layer,
                                     std::size_t slot_count) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        if (!kernel_dimensions_fit({ rows, feature_count, layer, slot_count })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "kv store dimensions exceed the kernel uint range"));
        }

        const auto kernel_rows = static_cast<std::uint32_t>(rows);
        const auto kernel_feature_count = static_cast<std::uint32_t>(feature_count);
        const auto kernel_layer = static_cast<std::uint32_t>(layer);
        const auto kernel_slot_count = static_cast<std::uint32_t>(slot_count);

        return implementation_->dispatch("store_kv", [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:implementation_->store_kv_f32_pipeline];
            bind_buffers(encoder, 0, keys.implementation_->buffer, values.implementation_->buffer,
                         slot_mapping.implementation_->buffer, key_cache.implementation_->buffer,
                         value_cache.implementation_->buffer);
            bind_constants(encoder, 5, kernel_rows, kernel_feature_count, kernel_layer,
                           kernel_slot_count);

            [encoder dispatchThreads:MTLSizeMake(feature_count, rows, 1)
                threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                          implementation_->store_kv_f32_pipeline, feature_count,
                                          rows)];
        });
    }
}

result<void, metal_error>
metal_kernels::dispatch_paged_flash_attention_prefill_f32(const metal_buffer& queries,
                                                          const metal_buffer& positions,
                                                          const metal_buffer& block_table,
                                                          const metal_buffer& block_table_offsets,
                                                          const metal_buffer& block_table_lengths,
                                                          const metal_buffer& key_cache,
                                                          const metal_buffer& value_cache,
                                                          const metal_buffer& query_tile_starts,
                                                          const metal_buffer& query_tile_lengths,
                                                          metal_buffer& output,
                                                          std::size_t rows,
                                                          std::size_t query_head_count,
                                                          std::size_t kv_head_count,
                                                          std::size_t head_dimension,
                                                          std::size_t block_size,
                                                          std::size_t slot_count,
                                                          std::size_t layer,
                                                          std::size_t block_table_entry_count,
                                                          std::size_t query_tile_count) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        const auto simd_width = static_cast<std::size_t>(
            implementation_->paged_flash_attention_prefill_f32_pipeline.threadExecutionWidth);
        constexpr std::size_t features_per_lane = 4;
        if (!implementation_->flash_attention_enabled
            || head_dimension > simd_width * features_per_lane) {
            return dispatch_paged_attention_f32(
                queries, positions, block_table, block_table_offsets, block_table_lengths,
                key_cache, value_cache, output, rows, query_head_count, kv_head_count,
                head_dimension, block_size, slot_count, layer, block_table_entry_count);
        }

        if (!kernel_dimensions_fit({ rows, query_head_count, kv_head_count, head_dimension,
                                     block_size, slot_count, layer, block_table_entry_count,
                                     query_tile_count, simd_width })
            || query_tile_count == 0
            || query_tile_starts.size_bytes() < query_tile_count * sizeof(std::uint32_t)
            || query_tile_lengths.size_bytes() < query_tile_count * sizeof(std::uint32_t)) {
            return fail(make_error(metal_errc::invalid_input,
                                   "paged FlashAttention dimensions are invalid"));
        }

        const auto max_threads =
            static_cast<std::size_t>(implementation_->paged_flash_attention_prefill_f32_pipeline
                                         .maxTotalThreadsPerThreadgroup);
        if (simd_width == 0 || simd_width > max_threads) {
            return fail(
                make_error(metal_errc::invalid_input,
                           "paged FlashAttention SIMD width exceeds the threadgroup limit"));
        }

        const auto kernel_rows = static_cast<std::uint32_t>(rows);
        const auto kernel_query_head_count = static_cast<std::uint32_t>(query_head_count);
        const auto kernel_kv_head_count = static_cast<std::uint32_t>(kv_head_count);
        const auto kernel_head_dimension = static_cast<std::uint32_t>(head_dimension);
        const auto kernel_block_size = static_cast<std::uint32_t>(block_size);
        const auto kernel_slot_count = static_cast<std::uint32_t>(slot_count);
        const auto kernel_layer = static_cast<std::uint32_t>(layer);
        const auto kernel_block_table_entry_count =
            static_cast<std::uint32_t>(block_table_entry_count);
        const auto kernel_query_tile_count = static_cast<std::uint32_t>(query_tile_count);
        const auto kernel_simd_width = static_cast<std::uint32_t>(simd_width);

        return implementation_->dispatch(
            "paged_flash_attention_prefill", [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setComputePipelineState:implementation_->
                                                 paged_flash_attention_prefill_f32_pipeline];
                bind_buffers(
                    encoder, 0, queries.implementation_->buffer, positions.implementation_->buffer,
                    block_table.implementation_->buffer,
                    block_table_offsets.implementation_->buffer,
                    block_table_lengths.implementation_->buffer, key_cache.implementation_->buffer,
                    value_cache.implementation_->buffer, query_tile_starts.implementation_->buffer,
                    query_tile_lengths.implementation_->buffer, output.implementation_->buffer);
                bind_constants(encoder, 10, kernel_rows, kernel_query_head_count,
                               kernel_kv_head_count, kernel_head_dimension, kernel_block_size,
                               kernel_slot_count, kernel_layer, kernel_block_table_entry_count,
                               kernel_query_tile_count, kernel_simd_width);
                constexpr std::size_t query_tile_size = 8;
                constexpr std::size_t key_tile_size = 16;
                [encoder setThreadgroupMemoryLength:query_tile_size * key_tile_size * sizeof(float)
                                            atIndex:0];
                [encoder setThreadgroupMemoryLength:query_tile_size * sizeof(float) atIndex:1];
                [encoder setThreadgroupMemoryLength:query_tile_size * sizeof(float) atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(query_head_count, query_tile_count, 1)
                        threadsPerThreadgroup:MTLSizeMake(simd_width, 1, 1)];
            });
    }
}

result<void, metal_error>
metal_kernels::dispatch_paged_attention_f32(const metal_buffer& queries,
                                            const metal_buffer& positions,
                                            const metal_buffer& block_table,
                                            const metal_buffer& block_table_offsets,
                                            const metal_buffer& block_table_lengths,
                                            const metal_buffer& key_cache,
                                            const metal_buffer& value_cache,
                                            metal_buffer& output,
                                            std::size_t rows,
                                            std::size_t query_head_count,
                                            std::size_t kv_head_count,
                                            std::size_t head_dimension,
                                            std::size_t block_size,
                                            std::size_t slot_count,
                                            std::size_t layer,
                                            std::size_t block_table_entry_count) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        if (!kernel_dimensions_fit({ rows, query_head_count, kv_head_count, head_dimension,
                                     block_size, slot_count, layer, block_table_entry_count })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "paged attention dimensions exceed the kernel uint range"));
        }

        const auto max_threads = static_cast<std::size_t>(
            implementation_->paged_attention_f32_pipeline.maxTotalThreadsPerThreadgroup);
        if (head_dimension > max_threads) {
            return fail(make_error(metal_errc::invalid_input,
                                   "attention head dimension exceeds the threadgroup limit"));
        }

        const auto kernel_rows = static_cast<std::uint32_t>(rows);
        const auto kernel_query_head_count = static_cast<std::uint32_t>(query_head_count);
        const auto kernel_kv_head_count = static_cast<std::uint32_t>(kv_head_count);
        const auto kernel_head_dimension = static_cast<std::uint32_t>(head_dimension);
        const auto kernel_block_size = static_cast<std::uint32_t>(block_size);
        const auto kernel_slot_count = static_cast<std::uint32_t>(slot_count);
        const auto kernel_layer = static_cast<std::uint32_t>(layer);
        const auto kernel_block_table_entry_count =
            static_cast<std::uint32_t>(block_table_entry_count);
        const auto simd_width = static_cast<std::size_t>(
            implementation_->paged_attention_f32_pipeline.threadExecutionWidth);
        const auto simd_pipeline = implementation_->paged_attention_simd_f32_pipeline;
        const auto attention_simd_width =
            static_cast<std::size_t>(simd_pipeline.threadExecutionWidth);
        if (implementation_->simd_attention_enabled
            && rows > 1
            && head_dimension > 128
            && head_dimension <= 8 * attention_simd_width) {
            const auto kernel_simd_width = static_cast<std::uint32_t>(attention_simd_width);
            return implementation_->dispatch(
                "paged_attention_simd_prefill", [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setComputePipelineState:simd_pipeline];
                    bind_buffers(
                        encoder, 0, queries.implementation_->buffer,
                        positions.implementation_->buffer, block_table.implementation_->buffer,
                        block_table_offsets.implementation_->buffer,
                        block_table_lengths.implementation_->buffer,
                        key_cache.implementation_->buffer, value_cache.implementation_->buffer,
                        output.implementation_->buffer);
                    bind_constants(encoder, 8, kernel_rows, kernel_query_head_count,
                                   kernel_kv_head_count, kernel_head_dimension, kernel_block_size,
                                   kernel_slot_count, kernel_layer, kernel_block_table_entry_count,
                                   kernel_simd_width);
                    [encoder dispatchThreadgroups:MTLSizeMake(query_head_count, rows, 1)
                            threadsPerThreadgroup:MTLSizeMake(attention_simd_width, 1, 1)];
                });
        }
        const auto kernel_simdgroup_count =
            static_cast<std::uint32_t>((head_dimension - 1) / simd_width + 1);

        constexpr std::size_t attention_chunk_size = 64;
        constexpr std::uint32_t chunked_attention_min_tokens = 64;
        std::uint32_t decode_position = 0;
        if (rows == 1) {
            std::memcpy(&decode_position, positions.bytes().data(), sizeof(decode_position));
        }
        const auto chunk_count =
            rows == 1 ? (static_cast<std::size_t>(decode_position) / attention_chunk_size + 1) : 1;

        if (chunk_count > 1 && decode_position >= chunked_attention_min_tokens - 1) {
            const auto partial_stride = head_dimension + 2;
            constexpr auto max_size = std::numeric_limits<std::size_t>::max();
            if (query_head_count > max_size / chunk_count
                || query_head_count * chunk_count > max_size / partial_stride
                || query_head_count * chunk_count * partial_stride > max_size / sizeof(float)) {
                return fail(make_error(metal_errc::invalid_input,
                                       "paged attention partial buffer size overflows"));
            }
            const auto partial_size_bytes =
                query_head_count * chunk_count * partial_stride * sizeof(float);
            auto partials = context_.make_shared_buffer(partial_size_bytes);
            if (!partials) {
                return fail(partials.error());
            }

            const auto kernel_chunk_size = static_cast<std::uint32_t>(attention_chunk_size);
            const auto kernel_chunk_count = static_cast<std::uint32_t>(chunk_count);
            return implementation_->dispatch(
                "paged_attention_chunked_decode", [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setComputePipelineState:implementation_->
                                                     paged_attention_partial_f32_pipeline];
                    bind_buffers(
                        encoder, 0, queries.implementation_->buffer,
                        positions.implementation_->buffer, block_table.implementation_->buffer,
                        block_table_offsets.implementation_->buffer,
                        block_table_lengths.implementation_->buffer,
                        key_cache.implementation_->buffer, value_cache.implementation_->buffer,
                        partials->implementation_->buffer);
                    bind_constants(encoder, 8, kernel_rows, kernel_query_head_count,
                                   kernel_kv_head_count, kernel_head_dimension, kernel_block_size,
                                   kernel_slot_count, kernel_layer, kernel_block_table_entry_count,
                                   kernel_simdgroup_count, kernel_chunk_size, kernel_chunk_count);
                    [encoder setThreadgroupMemoryLength:head_dimension * sizeof(float) atIndex:0];
                    [encoder setThreadgroupMemoryLength:4 * sizeof(float) atIndex:1];
                    [encoder dispatchThreadgroups:MTLSizeMake(query_head_count * chunk_count, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(head_dimension, 1, 1)];

                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [encoder setComputePipelineState:implementation_->
                                                     paged_attention_reduce_f32_pipeline];
                    bind_buffers(encoder, 0, partials->implementation_->buffer,
                                 output.implementation_->buffer);
                    bind_constants(encoder, 2, kernel_rows, kernel_query_head_count,
                                   kernel_head_dimension, kernel_chunk_count);
                    [encoder setThreadgroupMemoryLength:chunk_count * sizeof(float) atIndex:0];
                    [encoder setThreadgroupMemoryLength:sizeof(float) atIndex:1];
                    [encoder dispatchThreadgroups:MTLSizeMake(query_head_count, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(head_dimension, 1, 1)];
                });
        }

        return implementation_->dispatch(
            rows == 1 ? "paged_attention_decode" : "paged_attention_prefill",
            [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setComputePipelineState:implementation_->paged_attention_f32_pipeline];
                bind_buffers(encoder, 0, queries.implementation_->buffer,
                             positions.implementation_->buffer, block_table.implementation_->buffer,
                             block_table_offsets.implementation_->buffer,
                             block_table_lengths.implementation_->buffer,
                             key_cache.implementation_->buffer, value_cache.implementation_->buffer,
                             output.implementation_->buffer);
                bind_constants(encoder, 8, kernel_rows, kernel_query_head_count,
                               kernel_kv_head_count, kernel_head_dimension, kernel_block_size,
                               kernel_slot_count, kernel_layer, kernel_block_table_entry_count,
                               kernel_simdgroup_count);
                [encoder setThreadgroupMemoryLength:head_dimension * sizeof(float) atIndex:0];
                [encoder setThreadgroupMemoryLength:4 * sizeof(float) atIndex:1];

                [encoder dispatchThreadgroups:MTLSizeMake(query_head_count, rows, 1)
                        threadsPerThreadgroup:MTLSizeMake(head_dimension, 1, 1)];
            });
    }
}

} // namespace chibillm
