#include "metal/metal_context.h"
#include "metal/metal_error.h"

#include <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace chibillm {
namespace {

metal_error
make_error(metal_errc code, std::string message)
{
    return metal_error {
        .code = code,
        .message = std::move(message),
    };
}

std::string
message_from_error(NSError* error, std::string_view fallback)
{
    if (error == nil) {
        return std::string(fallback);
    }

    const char* message = error.localizedDescription.UTF8String;
    return message == nullptr ? std::string(fallback) : std::string(message);
}

MTLSize
adaptive_2d_threadgroup_size(id<MTLComputePipelineState> pipeline,
                             std::size_t grid_width,
                             std::size_t grid_height)
{
    if (grid_height == 1) {
        const auto simd_width =
            static_cast<std::size_t>(pipeline.threadExecutionWidth);
        return MTLSizeMake(std::min(simd_width, grid_width), 1, 1);
    }

    constexpr std::size_t preferred_threadgroup_dimension = 16;
    const auto max_threads =
        static_cast<std::size_t>(pipeline.maxTotalThreadsPerThreadgroup);
    const auto threadgroup_width =
        std::min({ preferred_threadgroup_dimension, max_threads, grid_width });
    const auto threadgroup_height = std::min(
        { preferred_threadgroup_dimension, max_threads / threadgroup_width, grid_height });

    return MTLSizeMake(threadgroup_width, threadgroup_height, 1);
}

} // namespace

struct metal_buffer::implementation {
    id<MTLBuffer> buffer;
};

struct metal_context::implementation {
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> shader_library;
    id<MTLComputePipelineState> matmul_f32_pipeline;
    id<MTLComputePipelineState> linear_bf16_pipeline;
    id<MTLComputePipelineState> linear_split_bf16_pipeline;
    id<MTLComputePipelineState> embedding_bf16_pipeline;
    id<MTLComputePipelineState> rms_norm_bf16_pipeline;
    id<MTLComputePipelineState> silu_mul_f32_pipeline;
    id<MTLComputePipelineState> add_f32_pipeline;
    id<MTLComputePipelineState> rope_f32_pipeline;
    id<MTLComputePipelineState> store_kv_f32_pipeline;
    id<MTLComputePipelineState> paged_attention_f32_pipeline;
    // one open "pass" collects many dispatches into a single command buffer.
    // nil when no pass is open. Accessed by const dispatches through the shallow
    // constness of the owning unique_ptr (pointer constness, not pointee).
    id<MTLCommandBuffer> pass_command_buffer;
    id<MTLComputeCommandEncoder> pass_encoder;
    std::string device_name;

    // what one dispatch encodes into; see definition below.
    struct dispatch_frame;

    [[nodiscard]] result<dispatch_frame, metal_error> open_dispatch_encoder();
    [[nodiscard]] result<void, metal_error>
    complete_dispatch_encoder(const dispatch_frame& frame);
};

// what a dispatch needs to encode one kernel. command_buffer is nil when the
// dispatch is appending to an open compute pass instead of running standalone.
struct metal_context::implementation::dispatch_frame {
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
};

// returns the encoder a dispatch must encode into: the open pass's encoder when
// batching, or a fresh self-contained command buffer otherwise (legacy behavior).
result<metal_context::implementation::dispatch_frame, metal_error>
metal_context::implementation::open_dispatch_encoder()
{
    if (pass_encoder != nil) {
        return dispatch_frame { nil, pass_encoder };
    }

    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    if (command_buffer == nil) {
        return fail(make_error(metal_errc::command_buffer_creation_failed,
                               "failed to create a Metal command buffer"));
    }

    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return fail(make_error(metal_errc::command_encoder_creation_failed,
                               "failed to create a Metal compute encoder"));
    }

    return dispatch_frame { command_buffer, encoder };
}

// closes out one dispatch. Standalone frames commit and wait as before; kernels
// appended to an open pass are finished later by end_compute_pass().
result<void, metal_error>
metal_context::implementation::complete_dispatch_encoder(const dispatch_frame& frame)
{
    if (frame.command_buffer == nil) {
        return {};
    }

    [frame.encoder endEncoding];
    [frame.command_buffer commit];
    [frame.command_buffer waitUntilCompleted];

    if (frame.command_buffer.status != MTLCommandBufferStatusCompleted) {
        return fail(make_error(
            metal_errc::execution_failed,
            message_from_error(frame.command_buffer.error, "Metal command execution failed")));
    }

    return {};
}

metal_buffer::metal_buffer(std::unique_ptr<implementation> implementation) noexcept
    : implementation_(std::move(implementation))
{}

metal_buffer::metal_buffer(metal_buffer&&) noexcept = default;

metal_buffer& metal_buffer::operator=(metal_buffer&&) noexcept = default;

metal_buffer::~metal_buffer() = default;

std::size_t
metal_buffer::size_bytes() const noexcept
{
    return implementation_->buffer.length;
}

std::span<std::byte>
metal_buffer::bytes() noexcept
{
    return {
        static_cast<std::byte*>(implementation_->buffer.contents),
        size_bytes(),
    };
}

std::span<const std::byte>
metal_buffer::bytes() const noexcept
{
    return {
        static_cast<const std::byte*>(implementation_->buffer.contents),
        size_bytes(),
    };
}

result<metal_context, metal_error>
metal_context::make(std::string_view shader_source)
{
    @autoreleasepool {
        if (shader_source.empty()) {
            return fail(make_error(metal_errc::invalid_input, "shader source is empty"));
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return fail(make_error(metal_errc::no_device, "no Metal device is available"));
        }

        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        if (command_queue == nil) {
            return fail(make_error(metal_errc::command_queue_creation_failed,
                                   "failed to create the Metal command queue"));
        }

        NSString* source = [[NSString alloc] initWithBytes:shader_source.data()
                                                    length:shader_source.size()
                                                  encoding:NSUTF8StringEncoding];
        if (source == nil) {
            return fail(make_error(metal_errc::invalid_input, "shader source is not valid UTF-8"));
        }

        NSError* library_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:nil
                                                        error:&library_error];
        if (library == nil) {
            return fail(make_error(
                metal_errc::shader_library_creation_failed,
                message_from_error(library_error, "failed to compile the Metal shader library")));
        }

        id<MTLFunction> matmul_f32 = [library newFunctionWithName:@"matmul_f32"];
        if (matmul_f32 == nil) {
            return fail(make_error(metal_errc::shader_function_not_found,
                                   "the Metal shader library does not contain matmul_f32"));
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> matmul_f32_pipeline =
            [device newComputePipelineStateWithFunction:matmul_f32 error:&pipeline_error];
        if (matmul_f32_pipeline == nil) {
            return fail(make_error(
                metal_errc::pipeline_creation_failed,
                message_from_error(pipeline_error, "failed to create the matmul_f32 pipeline")));
        }

        id<MTLFunction> linear_bf16 = [library newFunctionWithName:@"linear_bf16"];
        if (linear_bf16 == nil) {
            return fail(make_error(metal_errc::shader_function_not_found,
                                   "the Metal shader library does not contain linear_bf16"));
        }

        pipeline_error = nil;
        id<MTLComputePipelineState> linear_bf16_pipeline =
            [device newComputePipelineStateWithFunction:linear_bf16 error:&pipeline_error];
        if (linear_bf16_pipeline == nil) {
            return fail(make_error(
                metal_errc::pipeline_creation_failed,
                message_from_error(pipeline_error, "failed to create the linear_bf16 pipeline")));
        }

        id<MTLFunction> linear_split_bf16 = [library newFunctionWithName:@"linear_split_bf16"];
        if (linear_split_bf16 == nil) {
            return fail(make_error(metal_errc::shader_function_not_found,
                                   "the Metal shader library does not contain linear_split_bf16"));
        }

        pipeline_error = nil;
        id<MTLComputePipelineState> linear_split_bf16_pipeline =
            [device newComputePipelineStateWithFunction:linear_split_bf16 error:&pipeline_error];
        if (linear_split_bf16_pipeline == nil) {
            return fail(
                make_error(metal_errc::pipeline_creation_failed,
                           message_from_error(pipeline_error,
                                              "failed to create the linear_split_bf16 pipeline")));
        }

        id<MTLFunction> embedding_bf16 = [library newFunctionWithName:@"embedding_bf16"];
        if (embedding_bf16 == nil) {
            return fail(make_error(metal_errc::shader_function_not_found,
                                   "the Metal shader library does not contain embedding_bf16"));
        }

        pipeline_error = nil;
        id<MTLComputePipelineState> embedding_bf16_pipeline =
            [device newComputePipelineStateWithFunction:embedding_bf16 error:&pipeline_error];
        if (embedding_bf16_pipeline == nil) {
            return fail(
                make_error(metal_errc::pipeline_creation_failed,
                           message_from_error(pipeline_error,
                                              "failed to create the embedding_bf16 pipeline")));
        }

        id<MTLFunction> rms_norm_bf16 = [library newFunctionWithName:@"rms_norm_bf16"];
        if (rms_norm_bf16 == nil) {
            return fail(make_error(metal_errc::shader_function_not_found,
                                   "the Metal shader library does not contain rms_norm_bf16"));
        }

        pipeline_error = nil;
        id<MTLComputePipelineState> rms_norm_bf16_pipeline =
            [device newComputePipelineStateWithFunction:rms_norm_bf16 error:&pipeline_error];
        if (rms_norm_bf16_pipeline == nil) {
            return fail(make_error(
                metal_errc::pipeline_creation_failed,
                message_from_error(pipeline_error, "failed to create the rms_norm_bf16 pipeline")));
        }

        id<MTLFunction> silu_mul_f32 = [library newFunctionWithName:@"silu_mul_f32"];
        if (silu_mul_f32 == nil) {
            return fail(make_error(metal_errc::shader_function_not_found,
                                   "the Metal shader library does not contain silu_mul_f32"));
        }

        pipeline_error = nil;
        id<MTLComputePipelineState> silu_mul_f32_pipeline =
            [device newComputePipelineStateWithFunction:silu_mul_f32 error:&pipeline_error];
        if (silu_mul_f32_pipeline == nil) {
            return fail(make_error(
                metal_errc::pipeline_creation_failed,
                message_from_error(pipeline_error, "failed to create the silu_mul_f32 pipeline")));
        }

        id<MTLFunction> add_f32 = [library newFunctionWithName:@"add_f32"];
        if (add_f32 == nil) {
            return fail(make_error(metal_errc::shader_function_not_found,
                                   "the Metal shader library does not contain add_f32"));
        }

        pipeline_error = nil;
        id<MTLComputePipelineState> add_f32_pipeline =
            [device newComputePipelineStateWithFunction:add_f32 error:&pipeline_error];
        if (add_f32_pipeline == nil) {
            return fail(make_error(
                metal_errc::pipeline_creation_failed,
                message_from_error(pipeline_error, "failed to create the add_f32 pipeline")));
        }

        id<MTLFunction> rope_f32 = [library newFunctionWithName:@"rope_f32"];
        if (rope_f32 == nil) {
            return fail(make_error(metal_errc::shader_function_not_found,
                                   "the Metal shader library does not contain rope_f32"));
        }

        pipeline_error = nil;
        id<MTLComputePipelineState> rope_f32_pipeline =
            [device newComputePipelineStateWithFunction:rope_f32 error:&pipeline_error];
        if (rope_f32_pipeline == nil) {
            return fail(make_error(
                metal_errc::pipeline_creation_failed,
                message_from_error(pipeline_error, "failed to create the rope_f32 pipeline")));
        }

        id<MTLFunction> store_kv_f32 = [library newFunctionWithName:@"store_kv_f32"];
        if (store_kv_f32 == nil) {
            return fail(make_error(metal_errc::shader_function_not_found,
                                   "the Metal shader library does not contain store_kv_f32"));
        }

        pipeline_error = nil;
        id<MTLComputePipelineState> store_kv_f32_pipeline =
            [device newComputePipelineStateWithFunction:store_kv_f32 error:&pipeline_error];
        if (store_kv_f32_pipeline == nil) {
            return fail(make_error(
                metal_errc::pipeline_creation_failed,
                message_from_error(pipeline_error, "failed to create the store_kv_f32 pipeline")));
        }

        id<MTLFunction> paged_attention_f32 = [library newFunctionWithName:@"paged_attention_f32"];
        if (paged_attention_f32 == nil) {
            return fail(
                make_error(metal_errc::shader_function_not_found,
                           "the Metal shader library does not contain paged_attention_f32"));
        }

        pipeline_error = nil;
        id<MTLComputePipelineState> paged_attention_f32_pipeline =
            [device newComputePipelineStateWithFunction:paged_attention_f32 error:&pipeline_error];
        if (paged_attention_f32_pipeline == nil) {
            return fail(make_error(
                metal_errc::pipeline_creation_failed,
                message_from_error(pipeline_error,
                                   "failed to create the paged_attention_f32 pipeline")));
        }

        const char* device_name = device.name.UTF8String;
        auto implementation = std::make_unique<metal_context::implementation>();
        implementation->device = device;
        implementation->command_queue = command_queue;
        implementation->shader_library = library;
        implementation->matmul_f32_pipeline = matmul_f32_pipeline;
        implementation->linear_bf16_pipeline = linear_bf16_pipeline;
        implementation->linear_split_bf16_pipeline = linear_split_bf16_pipeline;
        implementation->embedding_bf16_pipeline = embedding_bf16_pipeline;
        implementation->rms_norm_bf16_pipeline = rms_norm_bf16_pipeline;
        implementation->silu_mul_f32_pipeline = silu_mul_f32_pipeline;
        implementation->add_f32_pipeline = add_f32_pipeline;
        implementation->rope_f32_pipeline = rope_f32_pipeline;
        implementation->store_kv_f32_pipeline = store_kv_f32_pipeline;
        implementation->paged_attention_f32_pipeline = paged_attention_f32_pipeline;
        implementation->device_name = device_name == nullptr ? "unknown Metal device" : device_name;

        return metal_context { std::move(implementation) };
    }
}

metal_context::metal_context(std::unique_ptr<implementation> implementation) noexcept
    : implementation_(std::move(implementation))
{}

metal_context::metal_context(metal_context&&) noexcept = default;

metal_context& metal_context::operator=(metal_context&&) noexcept = default;

metal_context::~metal_context() = default;

std::string_view
metal_context::device_name() const noexcept
{
    return implementation_->device_name;
}

result<void, metal_error>
metal_context::begin_compute_pass()
{
    @autoreleasepool {
        if (implementation_->pass_encoder != nil) {
            return fail(make_error(metal_errc::invalid_input,
                                   "cannot begin a compute pass while one is already open"));
        }

        id<MTLCommandBuffer> command_buffer = [implementation_->command_queue commandBuffer];
        if (command_buffer == nil) {
            return fail(make_error(metal_errc::command_buffer_creation_failed,
                                   "failed to create a Metal command buffer"));
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return fail(make_error(metal_errc::command_encoder_creation_failed,
                                   "failed to create a Metal compute encoder"));
        }

        implementation_->pass_command_buffer = command_buffer;
        implementation_->pass_encoder = encoder;
        return {};
    }
}

result<void, metal_error>
metal_context::end_compute_pass()
{
    @autoreleasepool {
        if (implementation_->pass_encoder == nil || implementation_->pass_command_buffer == nil) {
            return fail(make_error(metal_errc::invalid_input,
                                   "cannot end a compute pass while none is open"));
        }

        // detach first so a failure below still leaves the context reusable.
        id<MTLCommandBuffer> command_buffer = implementation_->pass_command_buffer;
        id<MTLComputeCommandEncoder> encoder = implementation_->pass_encoder;
        implementation_->pass_command_buffer = nil;
        implementation_->pass_encoder = nil;

        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            return fail(make_error(
                metal_errc::execution_failed,
                message_from_error(command_buffer.error, "Metal command execution failed")));
        }

        return {};
    }
}

void
metal_context::abort_compute_pass() noexcept
{
    @autoreleasepool {
        if (implementation_->pass_command_buffer == nil) {
            return;
        }

        id<MTLCommandBuffer> command_buffer = implementation_->pass_command_buffer;
        id<MTLComputeCommandEncoder> encoder = implementation_->pass_encoder;
        implementation_->pass_command_buffer = nil;
        implementation_->pass_encoder = nil;

        // nothing encoded here may be read afterwards; commit only so shared-memory
        // writes settle deterministically before the caller sees the error.
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
}

result<metal_buffer, metal_error>
metal_context::make_shared_buffer(std::size_t size_bytes) const
{
    @autoreleasepool {
        if (size_bytes == 0) {
            return fail(make_error(metal_errc::invalid_input, "Metal buffer size must be nonzero"));
        }

        id<MTLBuffer> buffer =
            [implementation_->device newBufferWithLength:size_bytes
                                                 options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            return fail(make_error(metal_errc::buffer_creation_failed,
                                   "failed to create a shared Metal buffer"));
        }

        auto implementation = std::make_unique<metal_buffer::implementation>();
        implementation->buffer = buffer;
        return metal_buffer { std::move(implementation) };
    }
}

result<void, metal_error>
metal_context::dispatch_matmul(const metal_buffer& lhs,
                               const metal_buffer& rhs,
                               metal_buffer& output,
                               std::size_t rows,
                               std::size_t inner_dimension,
                               std::size_t columns) const
{
    @autoreleasepool {
        constexpr auto max_shader_dimension = std::numeric_limits<std::uint32_t>::max();
        if (rows > max_shader_dimension
            || inner_dimension > max_shader_dimension
            || columns > max_shader_dimension) {
            return fail(make_error(metal_errc::invalid_input,
                                   "matmul dimensions exceed the shader uint range"));
        }

        const auto shader_rows = static_cast<std::uint32_t>(rows);
        const auto shader_inner_dimension = static_cast<std::uint32_t>(inner_dimension);
        const auto shader_columns = static_cast<std::uint32_t>(columns);

        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;

        [encoder setComputePipelineState:implementation_->matmul_f32_pipeline];
        [encoder setBuffer:lhs.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:rhs.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:3];
        [encoder setBytes:&shader_inner_dimension length:sizeof(shader_inner_dimension) atIndex:4];
        [encoder setBytes:&shader_columns length:sizeof(shader_columns) atIndex:5];

        [encoder dispatchThreads:MTLSizeMake(columns, rows, 1)
            threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                      implementation_->matmul_f32_pipeline, columns, rows)];
        return implementation_->complete_dispatch_encoder(*opened);
    }
}

result<void, metal_error>
metal_context::dispatch_linear_bf16(const metal_buffer& input,
                                    const metal_buffer& weight,
                                    metal_buffer& output,
                                    std::size_t rows,
                                    std::size_t input_features,
                                    std::size_t output_features) const
{
    @autoreleasepool {
        constexpr auto max_shader_dimension = std::numeric_limits<std::uint32_t>::max();
        if (rows > max_shader_dimension
            || input_features > max_shader_dimension
            || output_features > max_shader_dimension) {
            return fail(make_error(metal_errc::invalid_input,
                                   "linear dimensions exceed the shader uint range"));
        }

        const auto shader_rows = static_cast<std::uint32_t>(rows);
        const auto shader_input_features = static_cast<std::uint32_t>(input_features);
        const auto shader_output_features = static_cast<std::uint32_t>(output_features);

        auto opened = implementation_->open_dispatch_encoder();
        if (!opened) {
            return fail(opened.error());
        }
        id<MTLComputeCommandEncoder> encoder = opened->encoder;

        [encoder setComputePipelineState:implementation_->linear_bf16_pipeline];
        [encoder setBuffer:input.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:weight.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:3];
        [encoder setBytes:&shader_input_features length:sizeof(shader_input_features) atIndex:4];
        [encoder setBytes:&shader_output_features length:sizeof(shader_output_features) atIndex:5];

        [encoder dispatchThreads:MTLSizeMake(output_features, rows, 1)
            threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                      implementation_->linear_bf16_pipeline,
                                      output_features,
                                      rows)];
        return implementation_->complete_dispatch_encoder(*opened);
    }
}

result<void, metal_error>
metal_context::dispatch_linear_split_bf16(const metal_buffer& input,
                                          const metal_buffer& weight,
                                          const std::array<metal_buffer*, 3>& outputs,
                                          std::size_t rows,
                                          std::size_t input_features,
                                          const std::array<std::size_t, 3>& widths) const
{
    @autoreleasepool {
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

        [encoder setComputePipelineState:implementation_->linear_split_bf16_pipeline];
        [encoder setBuffer:input.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:weight.implementation_->buffer offset:0 atIndex:1];
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            [encoder setBuffer:outputs[i]->implementation_->buffer offset:0 atIndex:2 + i];
        }
        [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:5];
        [encoder setBytes:&shader_input_features length:sizeof(shader_input_features) atIndex:6];
        for (std::size_t i = 0; i < shader_widths.size(); ++i) {
            [encoder setBytes:&shader_widths[i] length:sizeof(shader_widths[i]) atIndex:7 + i];
        }

        [encoder dispatchThreads:MTLSizeMake(total_width, rows, 1)
            threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                      implementation_->linear_split_bf16_pipeline,
                                      total_width,
                                      rows)];

        return implementation_->complete_dispatch_encoder(*opened);
    }
}

result<void, metal_error>
metal_context::dispatch_embedding_bf16(const metal_buffer& token_ids,
                                       const metal_buffer& weight,
                                       metal_buffer& output,
                                       std::size_t token_count,
                                       std::size_t hidden_size) const
{
    @autoreleasepool {
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
                                      implementation_->embedding_bf16_pipeline,
                                      hidden_size,
                                      token_count)];
        return implementation_->complete_dispatch_encoder(*opened);
    }
}

result<void, metal_error>
metal_context::dispatch_rms_norm_bf16(const metal_buffer& input,
                                      const metal_buffer& weight,
                                      metal_buffer& output,
                                      std::size_t rows,
                                      std::size_t hidden_size,
                                      float epsilon) const
{
    @autoreleasepool {
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
        return implementation_->complete_dispatch_encoder(*opened);
    }
}

result<void, metal_error>
metal_context::dispatch_silu_mul_f32(const metal_buffer& gate,
                                     const metal_buffer& up,
                                     metal_buffer& output,
                                     std::size_t element_count) const
{
    @autoreleasepool {
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
        return implementation_->complete_dispatch_encoder(*opened);
    }
}

result<void, metal_error>
metal_context::dispatch_add_f32(const metal_buffer& lhs,
                                const metal_buffer& rhs,
                                metal_buffer& output,
                                std::size_t element_count) const
{
    @autoreleasepool {
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
        return implementation_->complete_dispatch_encoder(*opened);
    }
}

result<void, metal_error>
metal_context::dispatch_rope_f32(const metal_buffer& input,
                                 const metal_buffer& positions,
                                 metal_buffer& output,
                                 std::size_t rows,
                                 std::size_t head_count,
                                 std::size_t head_dimension,
                                 float theta) const
{
    @autoreleasepool {
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
        [encoder setBytes:&theta length:sizeof(theta) atIndex:6];

        [encoder dispatchThreads:MTLSizeMake(pair_columns, rows, 1)
            threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                      implementation_->rope_f32_pipeline, pair_columns, rows)];
        return implementation_->complete_dispatch_encoder(*opened);
    }
}

result<void, metal_error>
metal_context::dispatch_store_kv_f32(const metal_buffer& keys,
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
                                      implementation_->store_kv_f32_pipeline,
                                      feature_count,
                                      rows)];
        return implementation_->complete_dispatch_encoder(*opened);
    }
}

result<void, metal_error>
metal_context::dispatch_paged_attention_f32(const metal_buffer& queries,
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
        [encoder setThreadgroupMemoryLength:head_dimension * sizeof(float) atIndex:0];
        [encoder setThreadgroupMemoryLength:4 * sizeof(float) atIndex:1];

        [encoder dispatchThreadgroups:MTLSizeMake(query_head_count, rows, 1)
                threadsPerThreadgroup:MTLSizeMake(head_dimension, 1, 1)];
        return implementation_->complete_dispatch_encoder(*opened);
    }
}

} // namespace chibillm
