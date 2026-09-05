#include "metal/metal_kernels.h"
#include "metal/metal_context_internal.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace chibillm {
namespace {
MTLSize
adaptive_2d_threadgroup_size(id<MTLComputePipelineState> pipeline,
                             std::size_t grid_width,
                             std::size_t grid_height)
{
    if (grid_height == 1) {
        const auto simd_width = static_cast<std::size_t>(pipeline.threadExecutionWidth);
        return MTLSizeMake(std::min(simd_width, grid_width), 1, 1);
    }

    constexpr std::size_t preferred_threadgroup_dimension = 16;
    const auto max_threads = static_cast<std::size_t>(pipeline.maxTotalThreadsPerThreadgroup);
    const auto threadgroup_width =
        std::min({ preferred_threadgroup_dimension, max_threads, grid_width });
    const auto threadgroup_height =
        std::min({ preferred_threadgroup_dimension, max_threads / threadgroup_width, grid_height });

    return MTLSizeMake(threadgroup_width, threadgroup_height, 1);
}

} // namespace

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
        constexpr auto max_shader_dimension = std::numeric_limits<std::uint32_t>::max();
        if (input_features > max_shader_dimension || output_features > max_shader_dimension) {
            return fail(make_error(metal_errc::invalid_input,
                                   "linear-add dimensions exceed the shader uint range"));
        }

        const auto shader_input_features = static_cast<std::uint32_t>(input_features);
        const auto shader_output_features = static_cast<std::uint32_t>(output_features);
        const auto pipeline = implementation_->linear_add_bf16_decode_pipeline;
        const auto simd_width = static_cast<std::size_t>(pipeline.threadExecutionWidth);
        constexpr std::size_t preferred_thread_count = 64;
        auto thread_count =
            std::min(preferred_thread_count,
                     static_cast<std::size_t>(pipeline.maxTotalThreadsPerThreadgroup));
        thread_count -= thread_count % simd_width;
        const auto outputs_per_threadgroup = thread_count / simd_width;
        const auto threadgroup_count = (output_features - 1) / outputs_per_threadgroup + 1;
        const auto shader_outputs_per_threadgroup =
            static_cast<std::uint32_t>(outputs_per_threadgroup);
        const auto shader_simd_width = static_cast<std::uint32_t>(simd_width);

        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:weight.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:residual.implementation_->buffer offset:0 atIndex:2];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:3];
        [encoder setBytes:&shader_input_features length:sizeof(shader_input_features) atIndex:4];
        [encoder setBytes:&shader_output_features length:sizeof(shader_output_features) atIndex:5];
        [encoder setBytes:&shader_outputs_per_threadgroup
                   length:sizeof(shader_outputs_per_threadgroup)
                  atIndex:6];
        [encoder setBytes:&shader_simd_width length:sizeof(shader_simd_width) atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroup_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
        const auto profile_name =
            input_features > output_features ? "linear_decode_contract" : "linear_decode_square";
        return implementation_->complete_dispatch_encoder(*opened, profile_name);
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
        constexpr auto max_shader_dimension = std::numeric_limits<std::uint32_t>::max();
        const auto total_width = widths[0] + widths[1] + widths[2];
        if (rows > max_shader_dimension
            || input_features > max_shader_dimension
            || total_width > max_shader_dimension) {
            return fail(make_error(metal_errc::invalid_input,
                                   "linear split dimensions exceed the shader uint range"));
        }

        const auto shader_rows = static_cast<std::uint32_t>(rows);
        const auto shader_input_features = static_cast<std::uint32_t>(input_features);
        const std::array shader_widths {
            static_cast<std::uint32_t>(widths[0]),
            static_cast<std::uint32_t>(widths[1]),
            static_cast<std::uint32_t>(widths[2]),
        };

        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;

        const bool use_tensorops = implementation_->tensorops_enabled && rows > 1;
        const auto pipeline = use_tensorops ? implementation_->linear_bf16_tensorops_pipeline
            : rows == 1                     ? implementation_->linear_split_bf16_decode_pipeline
                                            : implementation_->linear_split_bf16_pipeline;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.implementation_->buffer offset:0 atIndex:0];
        if (use_tensorops) {
            [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:3];
            [encoder setBytes:&shader_input_features
                       length:sizeof(shader_input_features)
                      atIndex:4];
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
                [encoder setBytes:&shader_widths[i] length:sizeof(shader_widths[i]) atIndex:5];
                [encoder
                     dispatchThreadgroups:MTLSizeMake((widths[i] + 63) / 64, (rows + 63) / 64, 1)
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
            const auto shader_outputs_per_threadgroup =
                static_cast<std::uint32_t>(outputs_per_threadgroup);
            const auto shader_simd_width = static_cast<std::uint32_t>(simd_width);

            [encoder setBytes:&shader_input_features
                       length:sizeof(shader_input_features)
                      atIndex:5];
            for (std::size_t i = 0; i < shader_widths.size(); ++i) {
                [encoder setBytes:&shader_widths[i] length:sizeof(shader_widths[i]) atIndex:6 + i];
            }
            [encoder setBytes:&shader_outputs_per_threadgroup
                       length:sizeof(shader_outputs_per_threadgroup)
                      atIndex:9];
            [encoder setBytes:&shader_simd_width length:sizeof(shader_simd_width) atIndex:10];
            [encoder dispatchThreadgroups:MTLSizeMake(threadgroup_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
        } else if (!use_tensorops) {
            [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:5];
            [encoder setBytes:&shader_input_features
                       length:sizeof(shader_input_features)
                      atIndex:6];
            for (std::size_t i = 0; i < shader_widths.size(); ++i) {
                [encoder setBytes:&shader_widths[i] length:sizeof(shader_widths[i]) atIndex:7 + i];
            }
            [encoder dispatchThreads:MTLSizeMake(total_width, rows, 1)
                threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                          implementation_->linear_split_bf16_pipeline, total_width,
                                          rows)];
        }

        const auto profile_name = use_tensorops ? "linear_split_tensorops"
            : rows != 1                         ? "linear_split_prefill"
            : total_width > input_features * 4  ? "linear_split_decode_wide"
                                                : "linear_split_decode_qkv";
        return implementation_->complete_dispatch_encoder(*opened, profile_name);
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
        constexpr auto max_shader_dimension = std::numeric_limits<std::uint32_t>::max();
        if (token_count > max_shader_dimension || hidden_size > max_shader_dimension) {
            return fail(make_error(metal_errc::invalid_input,
                                   "embedding dimensions exceed the shader uint range"));
        }

        const auto shader_token_count = static_cast<std::uint32_t>(token_count);
        const auto shader_hidden_size = static_cast<std::uint32_t>(hidden_size);

        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;

        [encoder setComputePipelineState:implementation_->embedding_bf16_pipeline];
        [encoder setBuffer:token_ids.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:weight.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_token_count length:sizeof(shader_token_count) atIndex:3];
        [encoder setBytes:&shader_hidden_size length:sizeof(shader_hidden_size) atIndex:4];

        [encoder dispatchThreads:MTLSizeMake(hidden_size, token_count, 1)
            threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                      implementation_->embedding_bf16_pipeline, hidden_size,
                                      token_count)];
        return implementation_->complete_dispatch_encoder(*opened, "embedding");
    }
}

result<void, metal_error>
metal_kernels::dispatch_rms_norm_bf16(const metal_buffer& input,
                                      const metal_buffer& weight,
                                      metal_buffer& output,
                                      std::size_t rows,
                                      std::size_t hidden_size,
                                      float epsilon) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        constexpr auto max_shader_dimension = std::numeric_limits<std::uint32_t>::max();
        if (rows > max_shader_dimension || hidden_size > max_shader_dimension) {
            return fail(make_error(metal_errc::invalid_input,
                                   "rms norm dimensions exceed the shader uint range"));
        }

        const auto shader_rows = static_cast<std::uint32_t>(rows);
        const auto shader_hidden_size = static_cast<std::uint32_t>(hidden_size);

        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;

        [encoder setComputePipelineState:implementation_->rms_norm_bf16_pipeline];
        [encoder setBuffer:input.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:weight.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:3];
        [encoder setBytes:&shader_hidden_size length:sizeof(shader_hidden_size) atIndex:4];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:5];

        constexpr std::size_t preferred_thread_count = 256;
        const auto max_threads = static_cast<std::size_t>(
            implementation_->rms_norm_bf16_pipeline.maxTotalThreadsPerThreadgroup);
        const auto thread_count = std::bit_floor(std::min(preferred_thread_count, max_threads));

        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
        const auto profile_name = hidden_size <= 256 ? "rms_norm_grouped" : "rms_norm_hidden";
        return implementation_->complete_dispatch_encoder(*opened, profile_name);
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
                                               float epsilon) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        constexpr auto max_shader_dimension = std::numeric_limits<std::uint32_t>::max();
        if (rows > max_shader_dimension
            || hidden_size > max_shader_dimension
            || vocabulary_size > max_shader_dimension
            || partial_count > max_shader_dimension) {
            return fail(make_error(metal_errc::invalid_input,
                                   "GPU sampling dimensions exceed the shader uint range"));
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

        const auto shader_rows = static_cast<std::uint32_t>(rows);
        const auto shader_hidden_size = static_cast<std::uint32_t>(hidden_size);
        const auto shader_vocabulary_size = static_cast<std::uint32_t>(vocabulary_size);
        const auto shader_partial_count = static_cast<std::uint32_t>(partial_count);
        const auto outputs_per_simdgroup =
            static_cast<std::uint32_t>(outputs_per_threadgroup / simdgroup_count);
        const auto shader_simd_width = static_cast<std::uint32_t>(simd_width);
        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;

        [encoder setComputePipelineState:implementation_->gather_rows_f32_pipeline];
        [encoder setBuffer:hidden_states.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:row_indices.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:normalized.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_hidden_size length:sizeof(shader_hidden_size) atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(hidden_size, rows, 1)
            threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                      implementation_->gather_rows_f32_pipeline, hidden_size,
                                      rows)];

        [encoder setComputePipelineState:implementation_->rms_norm_bf16_pipeline];
        [encoder setBuffer:normalized.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:norm_weight.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:normalized.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:3];
        [encoder setBytes:&shader_hidden_size length:sizeof(shader_hidden_size) atIndex:4];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];

        [encoder setComputePipelineState:projection_pipeline];
        [encoder setBuffer:normalized.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:vocabulary_weight.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:partial_maxima.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_hidden_size length:sizeof(shader_hidden_size) atIndex:3];
        [encoder setBytes:&shader_vocabulary_size length:sizeof(shader_vocabulary_size) atIndex:4];
        [encoder setBytes:&shader_partial_count length:sizeof(shader_partial_count) atIndex:5];
        [encoder setBytes:&outputs_per_simdgroup length:sizeof(outputs_per_simdgroup) atIndex:6];
        [encoder setBytes:&shader_simd_width length:sizeof(shader_simd_width) atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(partial_count, rows, 1)
                threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];

        [encoder setComputePipelineState:implementation_->reduce_argmax_pipeline];
        [encoder setBuffer:partial_maxima.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:token_ids.implementation_->buffer offset:0 atIndex:1];
        [encoder setBytes:&shader_partial_count length:sizeof(shader_partial_count) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];

        return implementation_->complete_dispatch_encoder(*opened, "greedy_vocabulary_argmax");
    }
}

result<void, metal_error>
metal_kernels::dispatch_silu_mul_f32(const metal_buffer& gate,
                                     const metal_buffer& up,
                                     metal_buffer& output,
                                     std::size_t element_count) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        if (element_count > std::numeric_limits<std::uint32_t>::max()) {
            return fail(make_error(metal_errc::invalid_input,
                                   "silu multiply element count exceeds the shader uint range"));
        }

        const auto shader_element_count = static_cast<std::uint32_t>(element_count);

        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;

        [encoder setComputePipelineState:implementation_->silu_mul_f32_pipeline];
        [encoder setBuffer:gate.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:up.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_element_count length:sizeof(shader_element_count) atIndex:3];

        constexpr std::size_t preferred_threadgroup_size = 256;
        const auto threadgroup_size =
            std::min(preferred_threadgroup_size,
                     static_cast<std::size_t>(
                         implementation_->silu_mul_f32_pipeline.maxTotalThreadsPerThreadgroup));

        [encoder dispatchThreads:MTLSizeMake(element_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
        return implementation_->complete_dispatch_encoder(*opened, "silu_mul");
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
        if (element_count > std::numeric_limits<std::uint32_t>::max()) {
            return fail(make_error(metal_errc::invalid_input,
                                   "add element count exceeds the shader uint range"));
        }

        const auto shader_element_count = static_cast<std::uint32_t>(element_count);

        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;

        [encoder setComputePipelineState:implementation_->add_f32_pipeline];
        [encoder setBuffer:lhs.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:rhs.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_element_count length:sizeof(shader_element_count) atIndex:3];

        constexpr std::size_t preferred_threadgroup_size = 256;
        const auto threadgroup_size =
            std::min(preferred_threadgroup_size,
                     static_cast<std::size_t>(
                         implementation_->add_f32_pipeline.maxTotalThreadsPerThreadgroup));

        [encoder dispatchThreads:MTLSizeMake(element_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_size, 1, 1)];
        return implementation_->complete_dispatch_encoder(*opened, "add");
    }
}

result<void, metal_error>
metal_kernels::dispatch_rope_f32(const metal_buffer& input,
                                 const metal_buffer& positions,
                                 metal_buffer& output,
                                 std::size_t rows,
                                 std::size_t head_count,
                                 std::size_t head_dimension,
                                 float theta) const
{
    @autoreleasepool {
        const auto& implementation_ = context_.implementation_;
        constexpr auto max_shader_dimension = std::numeric_limits<std::uint32_t>::max();
        const auto pair_columns = head_count * (head_dimension / 2);
        if (rows > max_shader_dimension
            || head_count > max_shader_dimension
            || head_dimension > max_shader_dimension
            || pair_columns > max_shader_dimension) {
            return fail(make_error(metal_errc::invalid_input,
                                   "rope dimensions exceed the shader uint range"));
        }

        const auto shader_rows = static_cast<std::uint32_t>(rows);
        const auto shader_head_count = static_cast<std::uint32_t>(head_count);
        const auto shader_head_dimension = static_cast<std::uint32_t>(head_dimension);

        const auto frequency_key =
            std::pair { shader_head_dimension, std::bit_cast<std::uint32_t>(theta) };
        id<MTLBuffer> frequency_buffer;
        {
            const std::scoped_lock lock(implementation_->rope_frequency_mutex);
            auto found = implementation_->rope_frequency_buffers.find(frequency_key);
            if (found == implementation_->rope_frequency_buffers.end()) {
                std::vector<float> frequencies(head_dimension / 2);
                for (std::size_t pair = 0; pair < frequencies.size(); ++pair) {
                    const auto exponent =
                        -2.0F * static_cast<float>(pair) / static_cast<float>(head_dimension);
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

        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;

        [encoder setComputePipelineState:implementation_->rope_f32_pipeline];
        [encoder setBuffer:input.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:positions.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:3];
        [encoder setBytes:&shader_head_count length:sizeof(shader_head_count) atIndex:4];
        [encoder setBytes:&shader_head_dimension length:sizeof(shader_head_dimension) atIndex:5];
        [encoder setBuffer:frequency_buffer offset:0 atIndex:6];

        [encoder dispatchThreads:MTLSizeMake(pair_columns, rows, 1)
            threadsPerThreadgroup:adaptive_2d_threadgroup_size(implementation_->rope_f32_pipeline,
                                                               pair_columns, rows)];
        return implementation_->complete_dispatch_encoder(*opened, "rope");
    }
}

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
        constexpr auto max_shader_dimension = std::numeric_limits<std::uint32_t>::max();
        if (rows > max_shader_dimension
            || feature_count > max_shader_dimension
            || layer > max_shader_dimension
            || slot_count > max_shader_dimension) {
            return fail(make_error(metal_errc::invalid_input,
                                   "kv store dimensions exceed the shader uint range"));
        }

        const auto shader_rows = static_cast<std::uint32_t>(rows);
        const auto shader_feature_count = static_cast<std::uint32_t>(feature_count);
        const auto shader_layer = static_cast<std::uint32_t>(layer);
        const auto shader_slot_count = static_cast<std::uint32_t>(slot_count);

        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;

        [encoder setComputePipelineState:implementation_->store_kv_f32_pipeline];
        [encoder setBuffer:keys.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:values.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:slot_mapping.implementation_->buffer offset:0 atIndex:2];
        [encoder setBuffer:key_cache.implementation_->buffer offset:0 atIndex:3];
        [encoder setBuffer:value_cache.implementation_->buffer offset:0 atIndex:4];
        [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:5];
        [encoder setBytes:&shader_feature_count length:sizeof(shader_feature_count) atIndex:6];
        [encoder setBytes:&shader_layer length:sizeof(shader_layer) atIndex:7];
        [encoder setBytes:&shader_slot_count length:sizeof(shader_slot_count) atIndex:8];

        [encoder dispatchThreads:MTLSizeMake(feature_count, rows, 1)
            threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                      implementation_->store_kv_f32_pipeline, feature_count, rows)];
        return implementation_->complete_dispatch_encoder(*opened, "store_kv");
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

        constexpr auto max_shader_dimension = std::numeric_limits<std::uint32_t>::max();
        if (rows > max_shader_dimension
            || query_head_count > max_shader_dimension
            || kv_head_count > max_shader_dimension
            || head_dimension > max_shader_dimension
            || block_size > max_shader_dimension
            || slot_count > max_shader_dimension
            || layer > max_shader_dimension
            || block_table_entry_count > max_shader_dimension
            || query_tile_count > max_shader_dimension
            || simd_width > max_shader_dimension
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

        const auto shader_rows = static_cast<std::uint32_t>(rows);
        const auto shader_query_head_count = static_cast<std::uint32_t>(query_head_count);
        const auto shader_kv_head_count = static_cast<std::uint32_t>(kv_head_count);
        const auto shader_head_dimension = static_cast<std::uint32_t>(head_dimension);
        const auto shader_block_size = static_cast<std::uint32_t>(block_size);
        const auto shader_slot_count = static_cast<std::uint32_t>(slot_count);
        const auto shader_layer = static_cast<std::uint32_t>(layer);
        const auto shader_block_table_entry_count =
            static_cast<std::uint32_t>(block_table_entry_count);
        const auto shader_query_tile_count = static_cast<std::uint32_t>(query_tile_count);
        const auto shader_simd_width = static_cast<std::uint32_t>(simd_width);

        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;
        [encoder
            setComputePipelineState:implementation_->paged_flash_attention_prefill_f32_pipeline];
        [encoder setBuffer:queries.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:positions.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:block_table.implementation_->buffer offset:0 atIndex:2];
        [encoder setBuffer:block_table_offsets.implementation_->buffer offset:0 atIndex:3];
        [encoder setBuffer:block_table_lengths.implementation_->buffer offset:0 atIndex:4];
        [encoder setBuffer:key_cache.implementation_->buffer offset:0 atIndex:5];
        [encoder setBuffer:value_cache.implementation_->buffer offset:0 atIndex:6];
        [encoder setBuffer:query_tile_starts.implementation_->buffer offset:0 atIndex:7];
        [encoder setBuffer:query_tile_lengths.implementation_->buffer offset:0 atIndex:8];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:9];
        [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:10];
        [encoder setBytes:&shader_query_head_count
                   length:sizeof(shader_query_head_count)
                  atIndex:11];
        [encoder setBytes:&shader_kv_head_count length:sizeof(shader_kv_head_count) atIndex:12];
        [encoder setBytes:&shader_head_dimension length:sizeof(shader_head_dimension) atIndex:13];
        [encoder setBytes:&shader_block_size length:sizeof(shader_block_size) atIndex:14];
        [encoder setBytes:&shader_slot_count length:sizeof(shader_slot_count) atIndex:15];
        [encoder setBytes:&shader_layer length:sizeof(shader_layer) atIndex:16];
        [encoder setBytes:&shader_block_table_entry_count
                   length:sizeof(shader_block_table_entry_count)
                  atIndex:17];
        [encoder setBytes:&shader_query_tile_count
                   length:sizeof(shader_query_tile_count)
                  atIndex:18];
        [encoder setBytes:&shader_simd_width length:sizeof(shader_simd_width) atIndex:19];
        constexpr std::size_t query_tile_size = 8;
        constexpr std::size_t key_tile_size = 16;
        [encoder setThreadgroupMemoryLength:query_tile_size * key_tile_size * sizeof(float)
                                    atIndex:0];
        [encoder setThreadgroupMemoryLength:query_tile_size * sizeof(float) atIndex:1];
        [encoder setThreadgroupMemoryLength:query_tile_size * sizeof(float) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(query_head_count, query_tile_count, 1)
                threadsPerThreadgroup:MTLSizeMake(simd_width, 1, 1)];
        return implementation_->complete_dispatch_encoder(*opened, "paged_flash_attention_prefill");
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
        constexpr auto max_shader_dimension = std::numeric_limits<std::uint32_t>::max();
        if (rows > max_shader_dimension
            || query_head_count > max_shader_dimension
            || kv_head_count > max_shader_dimension
            || head_dimension > max_shader_dimension
            || block_size > max_shader_dimension
            || slot_count > max_shader_dimension
            || layer > max_shader_dimension
            || block_table_entry_count > max_shader_dimension) {
            return fail(make_error(metal_errc::invalid_input,
                                   "paged attention dimensions exceed the shader uint range"));
        }

        const auto max_threads = static_cast<std::size_t>(
            implementation_->paged_attention_f32_pipeline.maxTotalThreadsPerThreadgroup);
        if (head_dimension > max_threads) {
            return fail(make_error(metal_errc::invalid_input,
                                   "attention head dimension exceeds the threadgroup limit"));
        }

        const auto shader_rows = static_cast<std::uint32_t>(rows);
        const auto shader_query_head_count = static_cast<std::uint32_t>(query_head_count);
        const auto shader_kv_head_count = static_cast<std::uint32_t>(kv_head_count);
        const auto shader_head_dimension = static_cast<std::uint32_t>(head_dimension);
        const auto shader_block_size = static_cast<std::uint32_t>(block_size);
        const auto shader_slot_count = static_cast<std::uint32_t>(slot_count);
        const auto shader_layer = static_cast<std::uint32_t>(layer);
        const auto shader_block_table_entry_count =
            static_cast<std::uint32_t>(block_table_entry_count);
        const auto simd_width = static_cast<std::size_t>(
            implementation_->paged_attention_f32_pipeline.threadExecutionWidth);
        const auto shader_simdgroup_count =
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

            const auto shader_chunk_size = static_cast<std::uint32_t>(attention_chunk_size);
            const auto shader_chunk_count = static_cast<std::uint32_t>(chunk_count);
            auto opened = implementation_->open_dispatch_encoder();
            if (!opened) {
                return fail(opened.error());
            }
            id<MTLComputeCommandEncoder> encoder = opened->encoder;

            [encoder setComputePipelineState:implementation_->paged_attention_partial_f32_pipeline];
            [encoder setBuffer:queries.implementation_->buffer offset:0 atIndex:0];
            [encoder setBuffer:positions.implementation_->buffer offset:0 atIndex:1];
            [encoder setBuffer:block_table.implementation_->buffer offset:0 atIndex:2];
            [encoder setBuffer:block_table_offsets.implementation_->buffer offset:0 atIndex:3];
            [encoder setBuffer:block_table_lengths.implementation_->buffer offset:0 atIndex:4];
            [encoder setBuffer:key_cache.implementation_->buffer offset:0 atIndex:5];
            [encoder setBuffer:value_cache.implementation_->buffer offset:0 atIndex:6];
            [encoder setBuffer:partials->implementation_->buffer offset:0 atIndex:7];
            [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:8];
            [encoder setBytes:&shader_query_head_count
                       length:sizeof(shader_query_head_count)
                      atIndex:9];
            [encoder setBytes:&shader_kv_head_count length:sizeof(shader_kv_head_count) atIndex:10];
            [encoder setBytes:&shader_head_dimension
                       length:sizeof(shader_head_dimension)
                      atIndex:11];
            [encoder setBytes:&shader_block_size length:sizeof(shader_block_size) atIndex:12];
            [encoder setBytes:&shader_slot_count length:sizeof(shader_slot_count) atIndex:13];
            [encoder setBytes:&shader_layer length:sizeof(shader_layer) atIndex:14];
            [encoder setBytes:&shader_block_table_entry_count
                       length:sizeof(shader_block_table_entry_count)
                      atIndex:15];
            [encoder setBytes:&shader_simdgroup_count
                       length:sizeof(shader_simdgroup_count)
                      atIndex:16];
            [encoder setBytes:&shader_chunk_size length:sizeof(shader_chunk_size) atIndex:17];
            [encoder setBytes:&shader_chunk_count length:sizeof(shader_chunk_count) atIndex:18];
            [encoder setThreadgroupMemoryLength:head_dimension * sizeof(float) atIndex:0];
            [encoder setThreadgroupMemoryLength:4 * sizeof(float) atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(query_head_count * chunk_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(head_dimension, 1, 1)];

            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [encoder setComputePipelineState:implementation_->paged_attention_reduce_f32_pipeline];
            [encoder setBuffer:partials->implementation_->buffer offset:0 atIndex:0];
            [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:1];
            [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:2];
            [encoder setBytes:&shader_query_head_count
                       length:sizeof(shader_query_head_count)
                      atIndex:3];
            [encoder setBytes:&shader_head_dimension
                       length:sizeof(shader_head_dimension)
                      atIndex:4];
            [encoder setBytes:&shader_chunk_count length:sizeof(shader_chunk_count) atIndex:5];
            [encoder setThreadgroupMemoryLength:chunk_count * sizeof(float) atIndex:0];
            [encoder setThreadgroupMemoryLength:sizeof(float) atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake(query_head_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(head_dimension, 1, 1)];
            return implementation_->complete_dispatch_encoder(*opened,
                                                              "paged_attention_chunked_decode");
        }

        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;

        [encoder setComputePipelineState:implementation_->paged_attention_f32_pipeline];
        [encoder setBuffer:queries.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:positions.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:block_table.implementation_->buffer offset:0 atIndex:2];
        [encoder setBuffer:block_table_offsets.implementation_->buffer offset:0 atIndex:3];
        [encoder setBuffer:block_table_lengths.implementation_->buffer offset:0 atIndex:4];
        [encoder setBuffer:key_cache.implementation_->buffer offset:0 atIndex:5];
        [encoder setBuffer:value_cache.implementation_->buffer offset:0 atIndex:6];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:7];
        [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:8];
        [encoder setBytes:&shader_query_head_count
                   length:sizeof(shader_query_head_count)
                  atIndex:9];
        [encoder setBytes:&shader_kv_head_count length:sizeof(shader_kv_head_count) atIndex:10];
        [encoder setBytes:&shader_head_dimension length:sizeof(shader_head_dimension) atIndex:11];
        [encoder setBytes:&shader_block_size length:sizeof(shader_block_size) atIndex:12];
        [encoder setBytes:&shader_slot_count length:sizeof(shader_slot_count) atIndex:13];
        [encoder setBytes:&shader_layer length:sizeof(shader_layer) atIndex:14];
        [encoder setBytes:&shader_block_table_entry_count
                   length:sizeof(shader_block_table_entry_count)
                  atIndex:15];
        [encoder setBytes:&shader_simdgroup_count length:sizeof(shader_simdgroup_count) atIndex:16];
        [encoder setThreadgroupMemoryLength:head_dimension * sizeof(float) atIndex:0];
        [encoder setThreadgroupMemoryLength:4 * sizeof(float) atIndex:1];

        [encoder dispatchThreadgroups:MTLSizeMake(query_head_count, rows, 1)
                threadsPerThreadgroup:MTLSizeMake(head_dimension, 1, 1)];
        return implementation_->complete_dispatch_encoder(
            *opened, rows == 1 ? "paged_attention_decode" : "paged_attention_prefill");
    }
}

} // namespace chibillm
