#include "metal/metal_context.h"
#include "metal/metal_error.h"

#include <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <string_view>
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

bool
environment_flag(const char* name, bool default_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }
    return std::string_view(value) != "0"
        && std::string_view(value) != "false"
        && std::string_view(value) != "off";
}

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

struct metal_buffer::implementation {
    id<MTLBuffer> buffer;
};

struct metal_context::implementation {
    struct profile_stats {
        std::size_t calls = 0;
        double gpu_seconds = 0.0;
    };

    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> shader_library;
    id<MTLComputePipelineState> matmul_f32_pipeline;
    id<MTLComputePipelineState> linear_bf16_pipeline;
    id<MTLComputePipelineState> linear_bf16_decode_pipeline;
    id<MTLComputePipelineState> linear_bf16_tensorops_pipeline;
    id<MTLComputePipelineState> linear_split_bf16_pipeline;
    id<MTLComputePipelineState> linear_split_bf16_decode_pipeline;
    id<MTLComputePipelineState> embedding_bf16_pipeline;
    id<MTLComputePipelineState> rms_norm_bf16_pipeline;
    id<MTLComputePipelineState> silu_mul_f32_pipeline;
    id<MTLComputePipelineState> add_f32_pipeline;
    id<MTLComputePipelineState> rope_f32_pipeline;
    id<MTLComputePipelineState> store_kv_f32_pipeline;
    id<MTLComputePipelineState> paged_attention_f32_pipeline;
    id<MTLComputePipelineState> paged_attention_partial_f32_pipeline;
    id<MTLComputePipelineState> paged_attention_reduce_f32_pipeline;
    // one open "pass" collects many dispatches into a single command buffer.
    // nil when no pass is open. Accessed by const dispatches through the shallow
    // constness of the owning unique_ptr (pointer constness, not pointee).
    id<MTLCommandBuffer> pass_command_buffer;
    id<MTLComputeCommandEncoder> pass_encoder;
    std::string device_name;
    bool profiling_enabled = false;
    bool tensorops_enabled = false;
    std::map<std::string, profile_stats> profile;

    // what one dispatch encodes into; see definition below.
    struct dispatch_frame;

    [[nodiscard]] result<dispatch_frame, metal_error> open_dispatch_encoder();
    [[nodiscard]] result<void, metal_error>
    complete_dispatch_encoder(const dispatch_frame& frame, std::string_view profile_name = {});
    void dump_profile() const noexcept;
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
    if (pass_encoder != nil && !profiling_enabled) {
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
metal_context::implementation::complete_dispatch_encoder(const dispatch_frame& frame,
                                                         std::string_view profile_name)
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

    if (profiling_enabled && !profile_name.empty()) {
        auto& stats = profile[std::string(profile_name)];
        ++stats.calls;
        stats.gpu_seconds += frame.command_buffer.GPUEndTime - frame.command_buffer.GPUStartTime;
    }

    return {};
}

void
metal_context::implementation::dump_profile() const noexcept
{
    if (!profiling_enabled) {
        return;
    }
    for (const auto& [name, stats] : profile) {
        const auto total_ms = stats.gpu_seconds * 1000.0;
        std::fprintf(stderr, "[metal-profile] %-28s calls=%zu total_ms=%.3f avg_us=%.3f\n",
                     name.c_str(), stats.calls, total_ms, total_ms * 1000.0 / stats.calls);
    }
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
        id<MTLLibrary> library = nil;
        bool compiled_metal4 = false;
        const bool tensorops_requested = environment_flag("CHIBILLM_TENSOROPS", true);
        if (@available(macOS 26.0, *)) {
            if (tensorops_requested && [device supportsFamily:MTLGPUFamilyApple10]) {
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                options.languageVersion = MTLLanguageVersion4_0;
                options.preprocessorMacros = @{@"CHIBILLM_ENABLE_TENSOROPS" : @1};
                library = [device newLibraryWithSource:source options:options error:&library_error];
                compiled_metal4 = library != nil;
                if (library == nil && std::getenv("CHIBILLM_TENSOROPS") != nullptr) {
                    const char* error_message = library_error.localizedDescription.UTF8String;
                    std::fprintf(
                        stderr, "[metal] TensorOps shader unavailable, using legacy kernels: %s\n",
                        error_message == nullptr ? "unknown compilation error" : error_message);
                }
            }
        }
        if (library == nil) {
            library_error = nil;
            library = [device newLibraryWithSource:source options:nil error:&library_error];
        }
        if (library == nil) {
            const char* error_message = library_error.localizedDescription.UTF8String;
            std::fprintf(stderr, "[metal] failed to compile legacy shader library: %s\n",
                         error_message == nullptr ? "unknown compilation error" : error_message);
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

        id<MTLFunction> linear_bf16_decode = [library newFunctionWithName:@"linear_bf16_decode"];
        if (linear_bf16_decode == nil) {
            return fail(make_error(metal_errc::shader_function_not_found,
                                   "the Metal shader library does not contain linear_bf16_decode"));
        }

        pipeline_error = nil;
        id<MTLComputePipelineState> linear_bf16_decode_pipeline =
            [device newComputePipelineStateWithFunction:linear_bf16_decode error:&pipeline_error];
        if (linear_bf16_decode_pipeline == nil) {
            return fail(
                make_error(metal_errc::pipeline_creation_failed,
                           message_from_error(pipeline_error,
                                              "failed to create the linear_bf16_decode pipeline")));
        }

        id<MTLComputePipelineState> linear_bf16_tensorops_pipeline = nil;
        if (compiled_metal4) {
            id<MTLFunction> linear_bf16_tensorops =
                [library newFunctionWithName:@"linear_bf16_tensorops"];
            if (linear_bf16_tensorops != nil) {
                pipeline_error = nil;
                MTLComputePipelineDescriptor* descriptor =
                    [[MTLComputePipelineDescriptor alloc] init];
                descriptor.computeFunction = linear_bf16_tensorops;
                descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
                linear_bf16_tensorops_pipeline =
                    [device newComputePipelineStateWithDescriptor:descriptor
                                                          options:0
                                                       reflection:nil
                                                            error:&pipeline_error];
            }
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

        id<MTLFunction> linear_split_bf16_decode =
            [library newFunctionWithName:@"linear_split_bf16_decode"];
        if (linear_split_bf16_decode == nil) {
            return fail(
                make_error(metal_errc::shader_function_not_found,
                           "the Metal shader library does not contain linear_split_bf16_decode"));
        }

        pipeline_error = nil;
        id<MTLComputePipelineState> linear_split_bf16_decode_pipeline =
            [device newComputePipelineStateWithFunction:linear_split_bf16_decode
                                                  error:&pipeline_error];
        if (linear_split_bf16_decode_pipeline == nil) {
            return fail(make_error(
                metal_errc::pipeline_creation_failed,
                message_from_error(pipeline_error,
                                   "failed to create the linear_split_bf16_decode pipeline")));
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

        id<MTLFunction> paged_attention_partial_f32 =
            [library newFunctionWithName:@"paged_attention_partial_f32"];
        if (paged_attention_partial_f32 == nil) {
            return fail(make_error(
                metal_errc::shader_function_not_found,
                "the Metal shader library does not contain paged_attention_partial_f32"));
        }
        pipeline_error = nil;
        id<MTLComputePipelineState> paged_attention_partial_f32_pipeline =
            [device newComputePipelineStateWithFunction:paged_attention_partial_f32
                                                  error:&pipeline_error];
        if (paged_attention_partial_f32_pipeline == nil) {
            return fail(make_error(
                metal_errc::pipeline_creation_failed,
                message_from_error(pipeline_error,
                                   "failed to create paged_attention_partial_f32 pipeline")));
        }

        id<MTLFunction> paged_attention_reduce_f32 =
            [library newFunctionWithName:@"paged_attention_reduce_f32"];
        if (paged_attention_reduce_f32 == nil) {
            return fail(
                make_error(metal_errc::shader_function_not_found,
                           "the Metal shader library does not contain paged_attention_reduce_f32"));
        }
        pipeline_error = nil;
        id<MTLComputePipelineState> paged_attention_reduce_f32_pipeline =
            [device newComputePipelineStateWithFunction:paged_attention_reduce_f32
                                                  error:&pipeline_error];
        if (paged_attention_reduce_f32_pipeline == nil) {
            return fail(make_error(
                metal_errc::pipeline_creation_failed,
                message_from_error(pipeline_error,
                                   "failed to create paged_attention_reduce_f32 pipeline")));
        }

        const char* device_name = device.name.UTF8String;
        auto implementation = std::make_unique<metal_context::implementation>();
        implementation->device = device;
        implementation->command_queue = command_queue;
        implementation->shader_library = library;
        implementation->matmul_f32_pipeline = matmul_f32_pipeline;
        implementation->linear_bf16_pipeline = linear_bf16_pipeline;
        implementation->linear_bf16_decode_pipeline = linear_bf16_decode_pipeline;
        implementation->linear_bf16_tensorops_pipeline = linear_bf16_tensorops_pipeline;
        implementation->linear_split_bf16_pipeline = linear_split_bf16_pipeline;
        implementation->linear_split_bf16_decode_pipeline = linear_split_bf16_decode_pipeline;
        implementation->embedding_bf16_pipeline = embedding_bf16_pipeline;
        implementation->rms_norm_bf16_pipeline = rms_norm_bf16_pipeline;
        implementation->silu_mul_f32_pipeline = silu_mul_f32_pipeline;
        implementation->add_f32_pipeline = add_f32_pipeline;
        implementation->rope_f32_pipeline = rope_f32_pipeline;
        implementation->store_kv_f32_pipeline = store_kv_f32_pipeline;
        implementation->paged_attention_f32_pipeline = paged_attention_f32_pipeline;
        implementation->paged_attention_partial_f32_pipeline = paged_attention_partial_f32_pipeline;
        implementation->paged_attention_reduce_f32_pipeline = paged_attention_reduce_f32_pipeline;
        implementation->device_name = device_name == nullptr ? "unknown Metal device" : device_name;
        implementation->profiling_enabled = std::getenv("CHIBILLM_PROFILE_METAL") != nullptr;
        implementation->tensorops_enabled = linear_bf16_tensorops_pipeline != nil;

        return metal_context { std::move(implementation) };
    }
}

metal_context::metal_context(std::unique_ptr<implementation> implementation) noexcept
    : implementation_(std::move(implementation))
{}

metal_context::metal_context(metal_context&&) noexcept = default;

metal_context& metal_context::operator=(metal_context&&) noexcept = default;

metal_context::~metal_context()
{
    if (implementation_) {
        implementation_->dump_profile();
    }
}

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
            threadsPerThreadgroup:adaptive_2d_threadgroup_size(implementation_->matmul_f32_pipeline,
                                                               columns, rows)];
        return implementation_->complete_dispatch_encoder(*opened, "matmul");
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

        const bool use_tensorops = implementation_->tensorops_enabled && rows > 1;
        const auto pipeline = use_tensorops ? implementation_->linear_bf16_tensorops_pipeline
            : rows == 1                     ? implementation_->linear_bf16_decode_pipeline
                                            : implementation_->linear_bf16_pipeline;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input.implementation_->buffer offset:0 atIndex:0];
        [encoder setBuffer:weight.implementation_->buffer offset:0 atIndex:1];
        [encoder setBuffer:output.implementation_->buffer offset:0 atIndex:2];
        if (use_tensorops) {
            [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:3];
            [encoder setBytes:&shader_input_features
                       length:sizeof(shader_input_features)
                      atIndex:4];
            [encoder setBytes:&shader_output_features
                       length:sizeof(shader_output_features)
                      atIndex:5];
            const auto simd_width = static_cast<std::size_t>(pipeline.threadExecutionWidth);
            [encoder
                 dispatchThreadgroups:MTLSizeMake((output_features + 63) / 64, (rows + 63) / 64, 1)
                threadsPerThreadgroup:MTLSizeMake(4 * simd_width, 1, 1)];
        } else if (rows == 1) {
            const auto simd_width = static_cast<std::size_t>(pipeline.threadExecutionWidth);
            constexpr std::size_t preferred_thread_count = 256;
            auto thread_count =
                std::min(preferred_thread_count,
                         static_cast<std::size_t>(pipeline.maxTotalThreadsPerThreadgroup));
            thread_count -= thread_count % simd_width;
            const auto outputs_per_threadgroup = thread_count / simd_width;
            const auto threadgroup_count = (output_features - 1) / outputs_per_threadgroup + 1;
            const auto shader_outputs_per_threadgroup =
                static_cast<std::uint32_t>(outputs_per_threadgroup);
            const auto shader_simd_width = static_cast<std::uint32_t>(simd_width);

            [encoder setBytes:&shader_input_features
                       length:sizeof(shader_input_features)
                      atIndex:3];
            [encoder setBytes:&shader_output_features
                       length:sizeof(shader_output_features)
                      atIndex:4];
            [encoder setBytes:&shader_outputs_per_threadgroup
                       length:sizeof(shader_outputs_per_threadgroup)
                      atIndex:5];
            [encoder setBytes:&shader_simd_width length:sizeof(shader_simd_width) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(threadgroup_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
        } else {
            [encoder setBytes:&shader_rows length:sizeof(shader_rows) atIndex:3];
            [encoder setBytes:&shader_input_features
                       length:sizeof(shader_input_features)
                      atIndex:4];
            [encoder setBytes:&shader_output_features
                       length:sizeof(shader_output_features)
                      atIndex:5];
            [encoder dispatchThreads:MTLSizeMake(output_features, rows, 1)
                threadsPerThreadgroup:adaptive_2d_threadgroup_size(
                                          implementation_->linear_bf16_pipeline, output_features,
                                          rows)];
        }
        const auto profile_name = use_tensorops ? "linear_tensorops"
            : rows != 1                         ? "linear_prefill"
            : output_features > 10'000          ? "linear_decode_vocab"
            : input_features > output_features  ? "linear_decode_contract"
                                                : "linear_decode_square";
        return implementation_->complete_dispatch_encoder(*opened, profile_name);
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
            constexpr std::size_t preferred_thread_count = 256;
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
                                      implementation_->embedding_bf16_pipeline, hidden_size,
                                      token_count)];
        return implementation_->complete_dispatch_encoder(*opened, "embedding");
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
        const auto profile_name = hidden_size <= 256 ? "rms_norm_heads" : "rms_norm_hidden";
        return implementation_->complete_dispatch_encoder(*opened, profile_name);
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
        return implementation_->complete_dispatch_encoder(*opened, "silu_mul");
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
        return implementation_->complete_dispatch_encoder(*opened, "add");
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
            threadsPerThreadgroup:adaptive_2d_threadgroup_size(implementation_->rope_f32_pipeline,
                                                               pair_columns, rows)];
        return implementation_->complete_dispatch_encoder(*opened, "rope");
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
                                      implementation_->store_kv_f32_pipeline, feature_count, rows)];
        return implementation_->complete_dispatch_encoder(*opened, "store_kv");
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
        const auto simd_width = static_cast<std::size_t>(
            implementation_->paged_attention_f32_pipeline.threadExecutionWidth);
        const auto shader_simdgroup_count =
            static_cast<std::uint32_t>((head_dimension - 1) / simd_width + 1);

        constexpr std::size_t attention_chunk_size = 64;
        constexpr std::uint32_t chunked_attention_min_tokens = 128;
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
            auto partials = make_shared_buffer(partial_size_bytes);
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
