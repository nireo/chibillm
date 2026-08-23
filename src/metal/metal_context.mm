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
    id<MTLComputePipelineState> embedding_bf16_pipeline;
    id<MTLComputePipelineState> rms_norm_bf16_pipeline;
    id<MTLComputePipelineState> silu_mul_f32_pipeline;
    id<MTLComputePipelineState> add_f32_pipeline;
    id<MTLComputePipelineState> rope_f32_pipeline;
    std::string device_name;
};

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

        const char* device_name = device.name.UTF8String;
        auto implementation = std::make_unique<metal_context::implementation>();
        implementation->device = device;
        implementation->command_queue = command_queue;
        implementation->shader_library = library;
        implementation->matmul_f32_pipeline = matmul_f32_pipeline;
        implementation->linear_bf16_pipeline = linear_bf16_pipeline;
        implementation->embedding_bf16_pipeline = embedding_bf16_pipeline;
        implementation->rms_norm_bf16_pipeline = rms_norm_bf16_pipeline;
        implementation->silu_mul_f32_pipeline = silu_mul_f32_pipeline;
        implementation->add_f32_pipeline = add_f32_pipeline;
        implementation->rope_f32_pipeline = rope_f32_pipeline;
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

        [encoder setComputePipelineState:implementation_->matmul_f32_pipeline];
        [encoder setBuffer:lhs.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:rhs.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:3];
        [encoder setBytes:&shader_inner_dimension length:sizeof(shader_inner_dimension) atIndex:4];
        [encoder setBytes:&shader_columns length:sizeof(shader_columns) atIndex:5];

        constexpr std::size_t preferred_threadgroup_dimension = 16;
        const auto max_threads = static_cast<std::size_t>(
            implementation_->matmul_f32_pipeline.maxTotalThreadsPerThreadgroup);
        const auto threadgroup_width = std::min(preferred_threadgroup_dimension, max_threads);
        const auto threadgroup_height =
            std::min(preferred_threadgroup_dimension, max_threads / threadgroup_width);

        [encoder dispatchThreads:MTLSizeMake(columns, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_width, threadgroup_height, 1)];
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

        [encoder setComputePipelineState:implementation_->linear_bf16_pipeline];
        [encoder setBuffer:input.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:weight.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:3];
        [encoder setBytes:&shader_input_features length:sizeof(shader_input_features) atIndex:4];
        [encoder setBytes:&shader_output_features length:sizeof(shader_output_features) atIndex:5];

        constexpr std::size_t preferred_threadgroup_dimension = 16;
        const auto max_threads = static_cast<std::size_t>(
            implementation_->linear_bf16_pipeline.maxTotalThreadsPerThreadgroup);
        const auto threadgroup_width = std::min(preferred_threadgroup_dimension, max_threads);
        const auto threadgroup_height =
            std::min(preferred_threadgroup_dimension, max_threads / threadgroup_width);

        [encoder dispatchThreads:MTLSizeMake(output_features, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_width, threadgroup_height, 1)];
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

        [encoder setComputePipelineState:implementation_->embedding_bf16_pipeline];
        [encoder setBuffer:token_ids.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:weight.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_token_count length:sizeof(shader_token_count) atIndex:3];
        [encoder setBytes:&shader_hidden_size length:sizeof(shader_hidden_size) atIndex:4];

        constexpr std::size_t preferred_threadgroup_dimension = 16;
        const auto max_threads = static_cast<std::size_t>(
            implementation_->embedding_bf16_pipeline.maxTotalThreadsPerThreadgroup);
        const auto threadgroup_width = std::min(preferred_threadgroup_dimension, max_threads);
        const auto threadgroup_height =
            std::min(preferred_threadgroup_dimension, max_threads / threadgroup_width);

        [encoder dispatchThreads:MTLSizeMake(hidden_size, token_count, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_width, threadgroup_height, 1)];
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

        [encoder setComputePipelineState:implementation_->rope_f32_pipeline];
        [encoder setBuffer:input.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:positions.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];
        [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:3];
        [encoder setBytes:&shader_head_count length:sizeof(shader_head_count) atIndex:4];
        [encoder setBytes:&shader_head_dimension length:sizeof(shader_head_dimension) atIndex:5];
        [encoder setBytes:&theta length:sizeof(theta) atIndex:6];

        constexpr std::size_t preferred_threadgroup_dimension = 16;
        const auto max_threads = static_cast<std::size_t>(
            implementation_->rope_f32_pipeline.maxTotalThreadsPerThreadgroup);
        const auto threadgroup_width = std::min(preferred_threadgroup_dimension, max_threads);
        const auto threadgroup_height =
            std::min(preferred_threadgroup_dimension, max_threads / threadgroup_width);

        [encoder dispatchThreads:MTLSizeMake(pair_columns, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(threadgroup_width, threadgroup_height, 1)];
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

} // namespace chibillm
