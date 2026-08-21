#include "metal/metal_context.h"

#include <Metal/Metal.h>

#include <algorithm>
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
    id<MTLComputePipelineState> vector_add_pipeline;
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

        id<MTLFunction> vector_add = [library newFunctionWithName:@"vector_add"];
        if (vector_add == nil) {
            return fail(make_error(metal_errc::shader_function_not_found,
                                   "the Metal shader library does not contain vector_add"));
        }

        NSError* pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:vector_add error:&pipeline_error];
        if (pipeline == nil) {
            return fail(make_error(
                metal_errc::pipeline_creation_failed,
                message_from_error(pipeline_error, "failed to create the vector_add pipeline")));
        }

        const char* device_name = device.name.UTF8String;
        auto implementation = std::make_unique<metal_context::implementation>();
        implementation->device = device;
        implementation->command_queue = command_queue;
        implementation->shader_library = library;
        implementation->vector_add_pipeline = pipeline;
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
metal_context::dispatch_vector_add(const metal_buffer& lhs,
                                   const metal_buffer& rhs,
                                   metal_buffer& output,
                                   std::size_t element_count) const
{
    @autoreleasepool {
        if (element_count == 0 || element_count > std::numeric_limits<std::uint32_t>::max()) {
            return fail(make_error(metal_errc::invalid_input,
                                   "vector element count is outside the supported range"));
        }

        if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            return fail(make_error(metal_errc::invalid_input, "vector byte size overflows size_t"));
        }

        const auto required_bytes = element_count * sizeof(float);
        if (lhs.size_bytes() < required_bytes
            || rhs.size_bytes() < required_bytes
            || output.size_bytes() < required_bytes) {
            return fail(make_error(metal_errc::invalid_input,
                                   "a Metal buffer is too small for the vector dispatch"));
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

        [encoder setComputePipelineState:implementation_->vector_add_pipeline];
        [encoder setBuffer:lhs.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:rhs.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];

        const auto count = static_cast<std::uint32_t>(element_count);
        [encoder setBytes:&count length:sizeof(count) atIndex:3];

        constexpr std::size_t preferred_threadgroup_size = 256;
        const auto threadgroup_size =
            std::min(preferred_threadgroup_size,
                     static_cast<std::size_t>(
                         implementation_->vector_add_pipeline.maxTotalThreadsPerThreadgroup));
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

} // namespace chibillm
