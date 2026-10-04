#include "metal/metal_dispatch_internal.h"
#include "metal/metal_kernels.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace chibillm {
using namespace metal_dispatch;

result<void, metal_error>
metal_kernels::dispatch_linear_add_bf16(const metal_buffer& input,
                                        const metal_buffer& weight,
                                        const metal_buffer& residual,
                                        metal_buffer& output,
                                        std::size_t input_features,
                                        std::size_t output_features) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        if (!kernel_dimensions_fit({ input_features, output_features })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "linear-add dimensions exceed the kernel uint range"));
        }

        const auto kernel_input_features = static_cast<std::uint32_t>(input_features);
        const auto kernel_output_features = static_cast<std::uint32_t>(output_features);
        const auto pipeline = implementation_->linear_add_bf16_decode_pipeline;
        const auto simd_width = static_cast<std::size_t>(pipeline.threadExecutionWidth);
        constexpr std::size_t preferred_thread_count = 64;
        auto thread_count =
            std::min(preferred_thread_count,
                     static_cast<std::size_t>(pipeline.maxTotalThreadsPerThreadgroup));
        thread_count -= thread_count % simd_width;
        const auto outputs_per_threadgroup = thread_count / simd_width;
        const auto threadgroup_count = (output_features - 1) / outputs_per_threadgroup + 1;
        const auto kernel_outputs_per_threadgroup =
            static_cast<std::uint32_t>(outputs_per_threadgroup);
        const auto kernel_simd_width = static_cast<std::uint32_t>(simd_width);

        const auto profile_name =
            input_features > output_features ? "linear_decode_contract" : "linear_decode_square";
        return implementation_->dispatch(profile_name, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            bind_buffers(encoder, 0, input.implementation_->buffer, weight.implementation_->buffer,
                         residual.implementation_->buffer, output.implementation_->buffer);
            bind_constants(encoder, 4, kernel_input_features, kernel_output_features,
                           kernel_outputs_per_threadgroup, kernel_simd_width);
            [encoder dispatchThreadgroups:MTLSizeMake(threadgroup_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
        });
    }
}

result<void, metal_error>
metal_kernels::dispatch_linear_split_bf16(const metal_buffer& input,
                                          const metal_buffer& weight,
                                          const std::array<metal_buffer*, 3>& outputs,
                                          std::size_t rows,
                                          std::size_t input_features,
                                          const std::array<std::size_t, 3>& widths) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        const auto total_width = widths[0] + widths[1] + widths[2];
        if (!kernel_dimensions_fit({ rows, input_features, total_width })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "linear split dimensions exceed the kernel uint range"));
        }

        const auto kernel_rows = static_cast<std::uint32_t>(rows);
        const auto kernel_input_features = static_cast<std::uint32_t>(input_features);
        const std::array kernel_widths {
            static_cast<std::uint32_t>(widths[0]),
            static_cast<std::uint32_t>(widths[1]),
            static_cast<std::uint32_t>(widths[2]),
        };

        const bool use_tensorops = implementation_->tensorops_enabled && rows > 1;
        const auto profile_name = use_tensorops ? "linear_split_tensorops"
            : rows != 1                         ? "linear_split_prefill"
            : total_width > input_features * 4  ? "linear_split_decode_wide"
                                                : "linear_split_decode_qkv";
        return implementation_->dispatch(profile_name, [&](id<MTLComputeCommandEncoder> encoder) {
            const auto pipeline = use_tensorops ? implementation_->linear_bf16_tensorops_pipeline
                : rows == 1                     ? implementation_->linear_split_bf16_decode_pipeline
                                                : implementation_->linear_split_bf16_pipeline;
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input.implementation_->buffer offset:0 atIndex:0];
            if (use_tensorops) {
                bind_constants(encoder, 3, kernel_rows, kernel_input_features);
                const auto simd_width = static_cast<std::size_t>(pipeline.threadExecutionWidth);
                std::size_t weight_row_offset = 0;
                for (std::size_t i = 0; i < outputs.size(); ++i) {
                    if (widths[i] == 0) {
                        continue;
                    }
                    const auto weight_offset_bytes =
                        weight_row_offset * input_features * sizeof(std::uint16_t);
                    [encoder setBuffer:weight.implementation_->buffer
                                offset:weight_offset_bytes
                               atIndex:1];
                    [encoder setBuffer:outputs[i]->implementation_->buffer offset:0 atIndex:2];
                    bind_constants(encoder, 5, kernel_widths[i]);
                    [encoder dispatchThreadgroups:MTLSizeMake((widths[i] + 63) / 64,
                                                              (rows + 63) / 64, 1)
                            threadsPerThreadgroup:MTLSizeMake(4 * simd_width, 1, 1)];
                    weight_row_offset += widths[i];
                }
            } else {
                [encoder setBuffer:weight.implementation_->buffer offset:0 atIndex:1];
                for (std::size_t i = 0; i < outputs.size(); ++i) {
                    [encoder setBuffer:outputs[i]->implementation_->buffer offset:0 atIndex:2 + i];
                }
            }
            if (!use_tensorops && rows == 1) {
                const auto simd_width = static_cast<std::size_t>(pipeline.threadExecutionWidth);
                constexpr std::size_t preferred_thread_count = 64;
                auto thread_count =
                    std::min(preferred_thread_count,
                             static_cast<std::size_t>(pipeline.maxTotalThreadsPerThreadgroup));
                thread_count -= thread_count % simd_width;
                const auto outputs_per_threadgroup = thread_count / simd_width;
                const auto threadgroup_count = (total_width - 1) / outputs_per_threadgroup + 1;
                const auto kernel_outputs_per_threadgroup =
                    static_cast<std::uint32_t>(outputs_per_threadgroup);
                const auto kernel_simd_width = static_cast<std::uint32_t>(simd_width);

                bind_constants(encoder, 5, kernel_input_features);
                for (std::size_t i = 0; i < kernel_widths.size(); ++i) {
                    [encoder setBytes:&kernel_widths[i]
                               length:sizeof(kernel_widths[i])
                              atIndex:6 + i];
                }
                bind_constants(encoder, 9, kernel_outputs_per_threadgroup, kernel_simd_width);
                [encoder dispatchThreadgroups:MTLSizeMake(threadgroup_count, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
            } else if (!use_tensorops) {
                bind_constants(encoder, 5, kernel_rows, kernel_input_features);
                for (std::size_t i = 0; i < kernel_widths.size(); ++i) {
                    [encoder setBytes:&kernel_widths[i]
                               length:sizeof(kernel_widths[i])
                              atIndex:7 + i];
                }
                [encoder dispatchThreads:MTLSizeMake(total_width, rows, 1)
                    threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                              implementation_->linear_split_bf16_pipeline,
                                              total_width, rows)];
            }
        });
    }
}

} // namespace chibillm
