#include "metal/metal_kernels.h"
#include "metal/metal_context_internal.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

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

result<void, metal_error>
metal_kernels::dispatch_embedding_bf16(const metal_buffer& token_ids,
                                       const metal_buffer& weight,
                                       metal_buffer& output,
                                       std::size_t token_count,
                                       std::size_t hidden_size) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        if (!kernel_dimensions_fit({ token_count, hidden_size })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "embedding dimensions exceed the kernel uint range"));
        }

        const auto kernel_token_count = static_cast<std::uint32_t>(token_count);
        const auto kernel_hidden_size = static_cast<std::uint32_t>(hidden_size);

        return implementation_->dispatch("embedding", [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:implementation_->embedding_bf16_pipeline];
            bind_buffers(encoder, 0, token_ids.implementation_->buffer,
                         weight.implementation_->buffer, output.implementation_->buffer);
            bind_constants(encoder, 3, kernel_token_count, kernel_hidden_size);

            [encoder dispatchThreads:MTLSizeMake(hidden_size, token_count, 1)
                threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                          implementation_->embedding_bf16_pipeline, hidden_size,
                                          token_count)];
        });
    }
}

result<void, metal_error>
metal_kernels::dispatch_rms_norm_bf16(const metal_buffer& input,
                                      const metal_buffer& weight,
                                      metal_buffer& output,
                                      std::size_t rows,
                                      std::size_t hidden_size,
                                      float epsilon,
                                      bool zero_centered) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        if (!kernel_dimensions_fit({ rows, hidden_size })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "rms norm dimensions exceed the kernel uint range"));
        }

        const auto kernel_rows = static_cast<std::uint32_t>(rows);
        const auto kernel_hidden_size = static_cast<std::uint32_t>(hidden_size);

        const auto profile_name = hidden_size <= 256 ? "rms_norm_grouped" : "rms_norm_hidden";
        return implementation_->dispatch(profile_name, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:implementation_->rms_norm_bf16_pipeline];
            bind_buffers(encoder, 0, input.implementation_->buffer, weight.implementation_->buffer,
                         output.implementation_->buffer);
            bind_constants(encoder, 3, kernel_rows, kernel_hidden_size, epsilon);

            const float weight_offset = zero_centered ? 1.0F : 0.0F;
            bind_constants(encoder, 6, weight_offset);

            constexpr std::size_t preferred_thread_count = 256;
            const auto max_threads = static_cast<std::size_t>(
                implementation_->rms_norm_bf16_pipeline.maxTotalThreadsPerThreadgroup);
            const auto thread_count = std::bit_floor(std::min(preferred_thread_count, max_threads));

            [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
        });
    }
}

result<void, metal_error>
metal_kernels::dispatch_silu_mul_f32(const metal_buffer& gate,
                                     const metal_buffer& up,
                                     metal_buffer& output,
                                     std::size_t element_count,
                                     bool sigmoid_only) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        if (!kernel_dimensions_fit({ element_count })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "silu multiply element count exceeds the kernel uint range"));
        }

        const auto kernel_element_count = static_cast<std::uint32_t>(element_count);

        return implementation_->dispatch(
            sigmoid_only ? "sigmoid_mul" : "silu_mul", [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setComputePipelineState:implementation_->silu_mul_f32_pipeline];
                bind_buffers(encoder, 0, gate.implementation_->buffer, up.implementation_->buffer,
                             output.implementation_->buffer);
                bind_constants(encoder, 3, kernel_element_count);
                const std::uint32_t kernel_sigmoid_only = sigmoid_only;
                bind_constants(encoder, 4, kernel_sigmoid_only);

                constexpr std::size_t preferred_threadgroup_size = 256;
                const auto threadgroup_size = std::min(
                    preferred_threadgroup_size,
                    static_cast<std::size_t>(
                        implementation_->silu_mul_f32_pipeline.maxTotalThreadsPerThreadgroup));

                [encoder dispatchThreads:MTLSizeMake(element_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
            });
    }
}

result<void, metal_error>
metal_kernels::dispatch_add_f32(const metal_buffer& lhs,
                                const metal_buffer& rhs,
                                metal_buffer& output,
                                std::size_t element_count) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        if (!kernel_dimensions_fit({ element_count })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "add element count exceeds the kernel uint range"));
        }

        const auto kernel_element_count = static_cast<std::uint32_t>(element_count);

        return implementation_->dispatch("add", [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:implementation_->add_f32_pipeline];
            bind_buffers(encoder, 0, lhs.implementation_->buffer, rhs.implementation_->buffer,
                         output.implementation_->buffer);
            bind_constants(encoder, 3, kernel_element_count);

            constexpr std::size_t preferred_threadgroup_size = 256;
            const auto threadgroup_size =
                std::min(preferred_threadgroup_size,
                         static_cast<std::size_t>(
                             implementation_->add_f32_pipeline.maxTotalThreadsPerThreadgroup));

            [encoder dispatchThreads:MTLSizeMake(element_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
        });
    }
}

result<void, metal_error>
metal_kernels::dispatch_split_heads_f32(const metal_buffer& input,
                                        metal_buffer& first,
                                        metal_buffer& second,
                                        std::size_t rows,
                                        std::size_t head_count,
                                        std::size_t head_dimension) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        const auto width = head_count * head_dimension;
        if (!kernel_dimensions_fit({ rows, width })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "head split dimensions exceed the kernel uint range"));
        }

        return implementation_->dispatch("split_heads", [&](id<MTLComputeCommandEncoder> encoder) {
            const auto pipeline = implementation_->split_heads_f32_pipeline;
            const auto kernel_width = static_cast<std::uint32_t>(width);
            const auto kernel_head_dimension = static_cast<std::uint32_t>(head_dimension);
            [encoder setComputePipelineState:pipeline];
            bind_buffers(encoder, 0, input.implementation_->buffer, first.implementation_->buffer,
                         second.implementation_->buffer);
            bind_constants(encoder, 3, kernel_width, kernel_head_dimension);
            [encoder dispatchThreads:MTLSizeMake(width, rows, 1)
                threadsPerThreadgroup:adaptive_2d_threadgroup_size(pipeline, width, rows)];
        });
    }
}

result<void, metal_error>
metal_kernels::dispatch_rope_f32(const metal_buffer& input,
                                 const metal_buffer& positions,
                                 metal_buffer& output,
                                 std::size_t rows,
                                 std::size_t head_count,
                                 std::size_t head_dimension,
                                 float theta,
                                 std::size_t rotary_dimension) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        const auto pair_columns = head_count * (head_dimension / 2);
        if (!kernel_dimensions_fit({ rows, head_count, head_dimension, pair_columns })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "rope dimensions exceed the kernel uint range"));
        }

        const auto kernel_rows = static_cast<std::uint32_t>(rows);
        const auto kernel_head_count = static_cast<std::uint32_t>(head_count);
        const auto kernel_head_dimension = static_cast<std::uint32_t>(head_dimension);

        const auto kernel_rotary_dimension = static_cast<std::uint32_t>(rotary_dimension);
        const auto frequency_key =
            std::pair { kernel_rotary_dimension, std::bit_cast<std::uint32_t>(theta) };
        id<MTLBuffer> frequency_buffer;
        {
            const std::scoped_lock lock(implementation_->rope_frequency_mutex);
            auto found = implementation_->rope_frequency_buffers.find(frequency_key);
            if (found == implementation_->rope_frequency_buffers.end()) {
                std::vector<float> frequencies(rotary_dimension / 2);
                for (std::size_t pair = 0; pair < frequencies.size(); ++pair) {
                    const auto exponent =
                        -2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dimension);
                    frequencies[pair] = std::pow(theta, exponent);
                }
                frequency_buffer =
                    [implementation_->device newBufferWithBytes:frequencies.data()
                                                         length:frequencies.size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
                if (frequency_buffer == nil) {
                    return fail(make_error(metal_errc::buffer_creation_failed,
                                           "failed to allocate RoPE frequency buffer"));
                }
                implementation_->rope_frequency_buffers.emplace(frequency_key, frequency_buffer);
            } else {
                frequency_buffer = found->second;
            }
        }

        return implementation_->dispatch("rope", [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:implementation_->rope_f32_pipeline];
            bind_buffers(encoder, 0, input.implementation_->buffer,
                         positions.implementation_->buffer, output.implementation_->buffer);
            bind_constants(encoder, 3, kernel_rows, kernel_head_count, kernel_head_dimension);
            [encoder setBuffer:frequency_buffer offset:0 atIndex:6];
            bind_constants(encoder, 7, kernel_rotary_dimension);

            [encoder dispatchThreads:MTLSizeMake(pair_columns, rows, 1)
                threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                          implementation_->rope_f32_pipeline, pair_columns, rows)];
        });
    }
}

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
