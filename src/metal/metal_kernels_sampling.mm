#include "metal/metal_dispatch_internal.h"
#include "metal/metal_kernels.h"

#include <algorithm>
#include <cstdint>

namespace chibillm {
using namespace metal_dispatch;

result<void, metal_error>
metal_kernels::dispatch_greedy_vocabulary_bf16(const metal_buffer& hidden_states,
                                               const metal_buffer& row_indices,
                                               const metal_buffer& norm_weight,
                                               const metal_buffer& vocabulary_weight,
                                               metal_buffer& normalized,
                                               metal_buffer& partial_maxima,
                                               metal_buffer& token_ids,
                                               std::size_t rows,
                                               std::size_t hidden_size,
                                               std::size_t vocabulary_size,
                                               std::size_t partial_count,
                                               float epsilon,
                                               bool zero_centered) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        if (!kernel_dimensions_fit({ rows, hidden_size, vocabulary_size, partial_count })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "GPU sampling dimensions exceed the kernel uint range"));
        }

        constexpr std::size_t thread_count = 256;
        constexpr auto outputs_per_threadgroup = greedy_argmax_outputs_per_threadgroup;
        const auto projection_pipeline = implementation_->linear_bf16_partial_argmax_pipeline;
        const auto simd_width = static_cast<std::size_t>(projection_pipeline.threadExecutionWidth);
        const auto simdgroup_count = thread_count / simd_width;
        if (thread_count % simd_width != 0
            || outputs_per_threadgroup % simdgroup_count != 0
            || projection_pipeline.maxTotalThreadsPerThreadgroup < thread_count
            || implementation_->reduce_argmax_pipeline.maxTotalThreadsPerThreadgroup
                < thread_count) {
            return fail(make_error(metal_errc::invalid_input,
                                   "GPU sampling does not support this threadgroup layout"));
        }
        const auto expected_partials =
            (vocabulary_size + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
        if (partial_count != expected_partials) {
            return fail(make_error(metal_errc::invalid_input,
                                   "GPU sampling partial buffer has the wrong size"));
        }

        const auto kernel_rows = static_cast<std::uint32_t>(rows);
        const auto kernel_hidden_size = static_cast<std::uint32_t>(hidden_size);
        const auto kernel_vocabulary_size = static_cast<std::uint32_t>(vocabulary_size);
        const auto kernel_partial_count = static_cast<std::uint32_t>(partial_count);
        const auto outputs_per_simdgroup =
            static_cast<std::uint32_t>(outputs_per_threadgroup / simdgroup_count);
        const auto kernel_simd_width = static_cast<std::uint32_t>(simd_width);
        return implementation_->dispatch(
            "greedy_vocabulary_argmax", [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setComputePipelineState:implementation_->gather_rows_f32_pipeline];
                bind_buffers(encoder, 0, hidden_states.implementation_->buffer,
                             row_indices.implementation_->buffer,
                             normalized.implementation_->buffer);
                bind_constants(encoder, 3, kernel_hidden_size);
                [encoder dispatchThreads:MTLSizeMake(hidden_size, rows, 1)
                    threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                              implementation_->gather_rows_f32_pipeline,
                                              hidden_size, rows)];

                [encoder setComputePipelineState:implementation_->rms_norm_bf16_pipeline];
                bind_buffers(encoder, 0, normalized.implementation_->buffer,
                             norm_weight.implementation_->buffer,
                             normalized.implementation_->buffer);
                bind_constants(encoder, 3, kernel_rows, kernel_hidden_size, epsilon);

                const float weight_offset = zero_centered ? 1.0F : 0.0F;
                bind_constants(encoder, 6, weight_offset);

                [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];

                [encoder setComputePipelineState:projection_pipeline];
                bind_buffers(encoder, 0, normalized.implementation_->buffer,
                             vocabulary_weight.implementation_->buffer,
                             partial_maxima.implementation_->buffer);
                bind_constants(encoder, 3, kernel_hidden_size, kernel_vocabulary_size,
                               kernel_partial_count, outputs_per_simdgroup, kernel_simd_width);
                [encoder dispatchThreadgroups:MTLSizeMake(partial_count, rows, 1)
                        threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];

                [encoder setComputePipelineState:implementation_->reduce_argmax_pipeline];
                bind_buffers(encoder, 0, partial_maxima.implementation_->buffer,
                             token_ids.implementation_->buffer);
                bind_constants(encoder, 2, kernel_partial_count);
                [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
            });
    }
}

} // namespace chibillm
