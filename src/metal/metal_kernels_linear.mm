#include "metal/metal_context_internal.h"
#include "metal/metal_kernels.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

namespace chibillm {
using namespace metal_dispatch;

result<void, metal_error>
metal_kernels::dispatch_projection(const metal_buffer& input,
                                   matrix_view weight,
                                   const std::array<metal_buffer*, 3>& outputs,
                                   std::size_t rows,
                                   const std::array<std::size_t, 3>& widths,
                                   const metal_buffer* residual) const
{
    @autoreleasepool {
        if (!weight.supported())
            return fail(make_error(metal_errc::invalid_input, "unsupported matrix layout"));
        const auto input_features = weight.shape().dimensions()[1];
        const auto output_features = weight.shape().dimensions()[0];
        if (!kernel_dimensions_fit({ rows, input_features, output_features }, 1))
            return fail(make_error(metal_errc::invalid_input,
                                   "projection dimensions exceed the kernel uint range"));
        const auto& implementation_ = context_.implementation_;
        const auto* quantized = weight.quantized();
        const metal_buffer* dense = weight.dense() ? &weight.dense()->buffer() : nullptr;

        // Small batches benefit from direct matvecs. Large batches retain the
        // TensorOps path, amortizing one expansion over all activation rows.
        constexpr std::size_t expansion_batch_size = 8;
        if (quantized && implementation_->tensorops_enabled && rows >= expansion_batch_size) {
            const auto byte_count = weight.shape().element_count() * sizeof(std::uint16_t);
            auto& scratch = implementation_->projection_scratch;
            if (!scratch || scratch->size_bytes() < byte_count) {
                auto buffer = context_.make_shared_buffer(std::bit_ceil(byte_count));
                if (!buffer)
                    return fail(buffer.error());
                scratch.emplace(std::move(*buffer));
            }
            dense = &*scratch;
            const auto columns = static_cast<std::uint32_t>(input_features);
            CL_TRY(implementation_->dispatch(
                "expand_q4_bf16", [&](id<MTLComputeCommandEncoder> encoder) {
                    const auto pipeline = implementation_->expand_q4_bf16_pipeline;
                    [encoder setComputePipelineState:pipeline];
                    bind_buffers(encoder, 0, quantized->packed().buffer().implementation_->buffer,
                                 quantized->scales().buffer().implementation_->buffer,
                                 quantized->offsets().buffer().implementation_->buffer,
                                 dense->implementation_->buffer);
                    bind_constants(encoder, 4, columns);
                    [encoder dispatchThreads:MTLSizeMake(input_features, output_features, 1)
                        threadsPerThreadgroup:adaptive_2d_threadgroup_size(pipeline, input_features,
                                                                           output_features)];
                }));
        }
        if (dense) {
            if (residual && rows == 1)
                return dispatch_linear_add_bf16(input, *dense, *residual, *outputs[0],
                                                input_features, output_features);
            CL_TRY(
                dispatch_linear_split_bf16(input, *dense, outputs, rows, input_features, widths));
            if (residual)
                return dispatch_add_f32(*residual, *outputs[0], *outputs[0],
                                        rows * output_features);
            return {};
        }

        const auto pipeline = residual ? implementation_->linear_add_q4_decode_pipeline
                                       : implementation_->linear_split_q4_decode_pipeline;
        const auto simd_width = static_cast<std::size_t>(pipeline.threadExecutionWidth);
        auto threads = std::min(std::size_t(64),
                                static_cast<std::size_t>(pipeline.maxTotalThreadsPerThreadgroup));
        threads -= threads % simd_width;
        if (threads == 0)
            return fail(
                make_error(metal_errc::invalid_input, "unsupported projection threadgroup layout"));
        const auto outputs_per_group = threads / simd_width;
        const auto columns = static_cast<std::uint32_t>(input_features);
        const auto group_outputs = static_cast<std::uint32_t>(outputs_per_group);
        const auto lanes = static_cast<std::uint32_t>(simd_width);
        return implementation_->dispatch(
            residual ? "linear_q4_add" : "linear_q4_split",
            [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setComputePipelineState:pipeline];
                bind_buffers(encoder, 0, input.implementation_->buffer,
                             quantized->packed().buffer().implementation_->buffer);
                if (residual) {
                    const auto width = static_cast<std::uint32_t>(output_features);
                    bind_buffers(encoder, 2, residual->implementation_->buffer,
                                 outputs[0]->implementation_->buffer);
                    bind_constants(encoder, 4, columns, width, group_outputs, lanes);
                    bind_buffers(encoder, 8, quantized->scales().buffer().implementation_->buffer,
                                 quantized->offsets().buffer().implementation_->buffer);
                } else {
                    bind_buffers(encoder, 2, outputs[0]->implementation_->buffer,
                                 outputs[1]->implementation_->buffer,
                                 outputs[2]->implementation_->buffer);
                    const auto a = static_cast<std::uint32_t>(widths[0]);
                    const auto b = static_cast<std::uint32_t>(widths[1]);
                    const auto c = static_cast<std::uint32_t>(widths[2]);
                    bind_constants(encoder, 5, columns, a, b, c, group_outputs, lanes);
                    bind_buffers(encoder, 11, quantized->scales().buffer().implementation_->buffer,
                                 quantized->offsets().buffer().implementation_->buffer);
                }
                [encoder dispatchThreadgroups:MTLSizeMake((output_features + outputs_per_group - 1)
                                                              / outputs_per_group,
                                                          rows, 1)
                        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            });
    }
}

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
metal_kernels::dispatch_embedding(const metal_buffer& token_ids,
                                  matrix_view weight,
                                  metal_buffer& output,
                                  std::size_t token_count) const
{
    @autoreleasepool {
        if (!weight.supported())
            return fail(make_error(metal_errc::invalid_input, "unsupported matrix layout"));
        const auto hidden_size = weight.shape().dimensions()[1];
        const auto& implementation_ = context_.implementation_;
        const auto* quantized = weight.quantized();
        const auto& values = quantized ? quantized->packed().buffer() : weight.dense()->buffer();
        if (!kernel_dimensions_fit({ token_count, hidden_size })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "embedding dimensions exceed the kernel uint range"));
        }

        const auto kernel_token_count = static_cast<std::uint32_t>(token_count);
        const auto kernel_hidden_size = static_cast<std::uint32_t>(hidden_size);

        const auto pipeline = quantized ? implementation_->embedding_q4_pipeline
                                        : implementation_->embedding_bf16_pipeline;
        return implementation_->dispatch("embedding", [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setComputePipelineState:pipeline];
            bind_buffers(encoder, 0, token_ids.implementation_->buffer,
                         values.implementation_->buffer, output.implementation_->buffer);
            bind_constants(encoder, 3, kernel_token_count, kernel_hidden_size);
            if (quantized) {
                bind_buffers(encoder, 5, quantized->scales().buffer().implementation_->buffer,
                             quantized->offsets().buffer().implementation_->buffer);
            }

            [encoder dispatchThreads:MTLSizeMake(hidden_size, token_count, 1)
                threadsPerThreadgroup:adaptive_2d_threadgroup_size(pipeline, hidden_size,
                                                                   token_count)];
        });
    }
}

result<void, metal_error>
metal_kernels::dispatch_greedy_vocabulary(const metal_buffer& hidden_states,
                                          const metal_buffer& row_indices,
                                          const metal_buffer& norm_weight,
                                          matrix_view vocabulary_weight,
                                          metal_buffer& normalized,
                                          metal_buffer& partial_maxima,
                                          metal_buffer& token_ids,
                                          std::size_t rows,
                                          std::size_t partial_count,
                                          float epsilon,
                                          bool zero_centered) const
{
    @autoreleasepool {
        if (!vocabulary_weight.supported())
            return fail(make_error(metal_errc::invalid_input, "unsupported matrix layout"));
        const auto hidden_size = vocabulary_weight.shape().dimensions()[1];
        const auto vocabulary_size = vocabulary_weight.shape().dimensions()[0];
        const auto& implementation_ = context_.implementation_;
        const auto* quantized = vocabulary_weight.quantized();
        const auto& values =
            quantized ? quantized->packed().buffer() : vocabulary_weight.dense()->buffer();
        if (!kernel_dimensions_fit({ rows, hidden_size, vocabulary_size, partial_count })) {
            return fail(make_error(metal_errc::invalid_input,
                                   "GPU sampling dimensions exceed the kernel uint range"));
        }

        constexpr std::size_t thread_count = 256;
        constexpr auto outputs_per_threadgroup = greedy_argmax_outputs_per_threadgroup;
        const auto projection_pipeline = quantized
            ? implementation_->linear_q4_partial_argmax_pipeline
            : implementation_->linear_bf16_partial_argmax_pipeline;
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
                             values.implementation_->buffer,
                             partial_maxima.implementation_->buffer);
                bind_constants(encoder, 3, kernel_hidden_size, kernel_vocabulary_size,
                               kernel_partial_count, outputs_per_simdgroup, kernel_simd_width);
                if (quantized) {
                    bind_buffers(encoder, 8, quantized->scales().buffer().implementation_->buffer,
                                 quantized->offsets().buffer().implementation_->buffer);
                }
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
