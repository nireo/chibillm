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

} // namespace chibillm
