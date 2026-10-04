#include "metal/metal_dispatch_internal.h"
#include "metal/metal_kernels.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace chibillm {
using namespace metal_dispatch;

namespace {
bool
f32_rows_fit(const metal_buffer& buffer,
             std::size_t row_offset,
             std::size_t rows,
             std::size_t width)
{
    // Width is nonzero after geometry validation. Divide before forming byte offsets.
    const auto available_rows = buffer.size_bytes() / sizeof(float) / width;
    return row_offset <= available_rows && rows <= available_rows - row_offset;
}

} // namespace

result<void, metal_error>
metal_kernels::dispatch_causal_conv1d_silu(const metal_buffer& input,
                                           const metal_buffer& weight,
                                           metal_buffer& history,
                                           metal_buffer& output,
                                           std::size_t rows,
                                           std::size_t channels,
                                           std::size_t kernel,
                                           std::size_t row_offset) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        if (!kernel_dimensions_fit({ rows, channels, kernel }, 1))
            return fail(
                make_error(metal_errc::invalid_input, "invalid DeltaNet kernel dimensions"));
        if (!f32_rows_fit(input, row_offset, rows, channels)
            || !f32_rows_fit(output, row_offset, rows, channels))
            return fail(make_error(metal_errc::invalid_input, "DeltaNet row range exceeds buffer"));
        const auto row_offset_bytes = row_offset * channels * sizeof(float);
        const std::uint32_t geometry[] = { static_cast<std::uint32_t>(rows),
                                           static_cast<std::uint32_t>(channels),
                                           static_cast<std::uint32_t>(kernel) };
        const auto pipeline = implementation_->causal_conv1d_silu_pipeline;
        return implementation_->dispatch(
            "causal_conv1d_silu", [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:input.implementation_->buffer offset:row_offset_bytes atIndex:0];
                bind_buffers(encoder, 1, weight.implementation_->buffer,
                             history.implementation_->buffer);
                [encoder setBuffer:output.implementation_->buffer
                            offset:row_offset_bytes
                           atIndex:3];
                bind_constants(encoder, 4, geometry);
                const auto threads =
                    std::min<std::size_t>(256, pipeline.maxTotalThreadsPerThreadgroup);
                [encoder dispatchThreads:MTLSizeMake(channels, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            });
    }
}

result<void, metal_error>
metal_kernels::dispatch_gated_delta_rule(const metal_buffer& qkv,
                                         const metal_buffer& a,
                                         const metal_buffer& b,
                                         const metal_buffer& A_log,
                                         const metal_buffer& dt_bias,
                                         metal_buffer& state,
                                         metal_buffer& output,
                                         std::size_t rows,
                                         std::size_t key_heads,
                                         std::size_t value_heads,
                                         std::size_t key_dim,
                                         std::size_t value_dim,
                                         float epsilon,
                                         std::size_t row_offset,
                                         bool use_chunkwise) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        constexpr auto limit = std::numeric_limits<std::uint32_t>::max();
        if (!kernel_dimensions_fit({ rows, key_heads, value_heads, key_dim, value_dim }, 1)
            || value_heads % key_heads != 0
            || value_heads > limit / value_dim
            || key_heads > (limit - value_heads * value_dim) / key_dim / 2)
            return fail(
                make_error(metal_errc::invalid_input, "invalid DeltaNet kernel dimensions"));
        const auto value_width = value_heads * value_dim;
        const auto qkv_width = 2 * key_heads * key_dim + value_width;
        if (!f32_rows_fit(qkv, row_offset, rows, qkv_width)
            || !f32_rows_fit(a, row_offset, rows, value_heads)
            || !f32_rows_fit(b, row_offset, rows, value_heads)
            || !f32_rows_fit(output, row_offset, rows, value_width))
            return fail(make_error(metal_errc::invalid_input, "DeltaNet row range exceeds buffer"));
        // Allocate bounded scratch before encoding any state mutation. Metal's
        // serial compute encoder orders dispatches, including scratch reuse in
        // successive blocks; arena buffers stay alive until the pass completes.
        constexpr std::size_t block = 32;
        if (use_chunkwise && implementation_->chunkwise_delta_enabled && rows >= block) {
            auto normalized =
                context_.make_shared_buffer(block * 2 * key_heads * key_dim * sizeof(float));
            if (!normalized)
                return fail(normalized.error());
            auto gates = context_.make_shared_buffer(block * value_heads * 3 * sizeof(float));
            if (!gates)
                return fail(gates.error());
            auto products =
                context_.make_shared_buffer(block * block * value_heads * 3 * sizeof(float));
            if (!products)
                return fail(products.error());
            auto work = context_.make_shared_buffer(block * value_width * 3 * sizeof(float));
            if (!work)
                return fail(work.error());
            return implementation_->dispatch(
                "gated_delta_rule_chunkwise", [&](id<MTLComputeCommandEncoder> encoder) {
                    bind_buffers(encoder, 3, A_log.implementation_->buffer,
                                 dt_bias.implementation_->buffer, state.implementation_->buffer);
                    bind_constants(encoder, 8, epsilon);
                    bind_buffers(encoder, 9, normalized->implementation_->buffer,
                                 gates->implementation_->buffer, products->implementation_->buffer,
                                 work->implementation_->buffer);
                    for (std::size_t begin = 0; begin < rows; begin += block) {
                        const auto count = std::min(block, rows - begin);
                        const auto offset = row_offset + begin;
                        const std::uint32_t geometry[] = { static_cast<std::uint32_t>(count),
                                                           static_cast<std::uint32_t>(key_heads),
                                                           static_cast<std::uint32_t>(value_heads),
                                                           static_cast<std::uint32_t>(key_dim),
                                                           static_cast<std::uint32_t>(value_dim) };
                        bind_constants(encoder, 7, geometry);
                        [encoder setBuffer:qkv.implementation_->buffer
                                    offset:offset * qkv_width * sizeof(float)
                                   atIndex:0];
                        [encoder setBuffer:a.implementation_->buffer
                                    offset:offset * value_heads * sizeof(float)
                                   atIndex:1];
                        [encoder setBuffer:b.implementation_->buffer
                                    offset:offset * value_heads * sizeof(float)
                                   atIndex:2];
                        [encoder setBuffer:output.implementation_->buffer
                                    offset:offset * value_width * sizeof(float)
                                   atIndex:6];
                        auto dispatch = [&](id<MTLComputePipelineState> pipeline, MTLSize grid,
                                            MTLSize threads) {
                            [encoder setComputePipelineState:pipeline];
                            [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
                        };
                        dispatch(implementation_->delta_prepare_pipeline,
                                 MTLSizeMake(count, value_heads, 1), MTLSizeMake(32, 1, 1));
                        dispatch(implementation_->delta_products_pipeline,
                                 MTLSizeMake(count, count, value_heads), MTLSizeMake(8, 8, 1));
                        dispatch(implementation_->delta_project_pipeline,
                                 MTLSizeMake(value_width, count, 1), MTLSizeMake(32, 4, 1));
                        dispatch(implementation_->delta_solve_pipeline,
                                 MTLSizeMake(value_width, 1, 1), MTLSizeMake(128, 1, 1));
                        dispatch(implementation_->delta_finish_pipeline,
                                 MTLSizeMake(value_width, std::max(count, key_dim), 1),
                                 MTLSizeMake(32, 4, 1));
                    }
                });
        }
        const auto qkv_offset_bytes = row_offset * qkv_width * sizeof(float);
        const auto gate_offset_bytes = row_offset * value_heads * sizeof(float);
        const auto output_offset_bytes = row_offset * value_width * sizeof(float);
        const std::uint32_t geometry[] = { static_cast<std::uint32_t>(rows),
                                           static_cast<std::uint32_t>(key_heads),
                                           static_cast<std::uint32_t>(value_heads),
                                           static_cast<std::uint32_t>(key_dim),
                                           static_cast<std::uint32_t>(value_dim) };
        const auto pipeline = implementation_->gated_delta_rule_pipeline;
        return implementation_->dispatch(
            "gated_delta_rule", [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:qkv.implementation_->buffer offset:qkv_offset_bytes atIndex:0];
                [encoder setBuffer:a.implementation_->buffer offset:gate_offset_bytes atIndex:1];
                [encoder setBuffer:b.implementation_->buffer offset:gate_offset_bytes atIndex:2];
                bind_buffers(encoder, 3, A_log.implementation_->buffer,
                             dt_bias.implementation_->buffer, state.implementation_->buffer);
                [encoder setBuffer:output.implementation_->buffer
                            offset:output_offset_bytes
                           atIndex:6];
                bind_constants(encoder, 7, geometry, epsilon);
                const auto threads =
                    std::min<std::size_t>(256, pipeline.maxTotalThreadsPerThreadgroup);
                [encoder dispatchThreads:MTLSizeMake(value_heads * value_dim, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            });
    }
}

result<void, metal_error>
metal_kernels::dispatch_rms_norm_gated(const metal_buffer& input,
                                       const metal_buffer& gate,
                                       const metal_buffer& weight,
                                       metal_buffer& output,
                                       std::size_t groups,
                                       std::size_t width,
                                       float epsilon) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        if (!kernel_dimensions_fit({ groups, width }, 1))
            return fail(
                make_error(metal_errc::invalid_input, "invalid DeltaNet kernel dimensions"));
        const std::uint32_t geometry[] = { static_cast<std::uint32_t>(groups),
                                           static_cast<std::uint32_t>(width) };
        const auto pipeline = implementation_->rms_norm_gated_pipeline;
        return implementation_->dispatch(
            "rms_norm_gated", [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setComputePipelineState:pipeline];
                bind_buffers(encoder, 0, input.implementation_->buffer,
                             gate.implementation_->buffer, weight.implementation_->buffer,
                             output.implementation_->buffer);
                bind_constants(encoder, 4, geometry, epsilon);
                const auto threads =
                    std::min<std::size_t>(256, pipeline.maxTotalThreadsPerThreadgroup);
                [encoder dispatchThreads:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            });
    }
}

} // namespace chibillm
