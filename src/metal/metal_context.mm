#include "metal/metal_context.h"
#include "metal/metal_context_internal.h"
#include "metal/metal_error.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

#include <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace chibillm {
namespace {

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
environment_flag(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr
        && std::string_view(value) != "0"
        && std::string_view(value) != "false"
        && std::string_view(value) != "off";
}

result<id<MTLComputePipelineState>, metal_error>
make_compute_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name)
{
    const char* utf8_name = name.UTF8String;
    const std::string function_name = utf8_name == nullptr ? "unknown" : utf8_name;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        return fail(make_error(metal_errc::shader_function_not_found,
                               "the Metal shader library does not contain " + function_name));
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&error];
    if (pipeline == nil) {
        return fail(make_error(
            metal_errc::pipeline_creation_failed,
            message_from_error(error, "failed to create the " + function_name + " pipeline")));
    }
    return pipeline;
}

} // namespace

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
    std::vector<std::pair<std::string_view, const profile_stats*>> sorted;
    sorted.reserve(profile.size());
    double total_seconds = 0.0;
    for (const auto& [name, stats] : profile) {
        sorted.emplace_back(name, &stats);
        total_seconds += stats.gpu_seconds;
    }
    std::ranges::sort(sorted, {}, [](const auto& entry) { return -entry.second->gpu_seconds; });
    for (const auto& [name, stats] : sorted) {
        const auto total_ms = stats->gpu_seconds * 1000.0;
        const auto percentage =
            total_seconds == 0.0 ? 0.0 : 100.0 * stats->gpu_seconds / total_seconds;
        std::fprintf(
            stderr, "[metal-profile] %-28.*s calls=%zu total_ms=%8.3f avg_us=%8.3f share=%5.1f%%\n",
            static_cast<int>(name.size()), name.data(), stats->calls, total_ms,
            total_ms * 1000.0 / stats->calls, percentage);
    }
    std::fprintf(stderr, "[metal-profile] %-28s total_ms=%8.3f\n", "all instrumented work",
                 total_seconds * 1000.0);
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
        if (@available(macOS 26.0, *)) {
            if ([device supportsFamily:MTLGPUFamilyApple10]) {
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                options.languageVersion = MTLLanguageVersion4_0;
                options.preprocessorMacros = @{@"CHIBILLM_ENABLE_TENSOROPS" : @1};
                library = [device newLibraryWithSource:source options:options error:&library_error];
                compiled_metal4 = library != nil;
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

        auto implementation = std::make_unique<metal_context::implementation>();
        CL_TRY_ASSIGN(implementation->linear_add_bf16_decode_pipeline,
                      make_compute_pipeline(device, library, @"linear_add_bf16_decode"));

        if (compiled_metal4) {
            id<MTLFunction> linear_bf16_tensorops =
                [library newFunctionWithName:@"linear_bf16_tensorops"];
            if (linear_bf16_tensorops != nil) {
                NSError* pipeline_error = nil;
                MTLComputePipelineDescriptor* descriptor =
                    [[MTLComputePipelineDescriptor alloc] init];
                descriptor.computeFunction = linear_bf16_tensorops;
                descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
                implementation->linear_bf16_tensorops_pipeline =
                    [device newComputePipelineStateWithDescriptor:descriptor
                                                          options:0
                                                       reflection:nil
                                                            error:&pipeline_error];
                if (implementation->linear_bf16_tensorops_pipeline == nil) {
                    std::fprintf(
                        stderr, "[metal] TensorOps pipeline unavailable: %s\n",
                        message_from_error(pipeline_error, "unknown pipeline error").c_str());
                }
            }
        }

        CL_TRY_ASSIGN(implementation->linear_split_bf16_pipeline,
                      make_compute_pipeline(device, library, @"linear_split_bf16"));
        CL_TRY_ASSIGN(implementation->linear_split_bf16_decode_pipeline,
                      make_compute_pipeline(device, library, @"linear_split_bf16_decode"));
        CL_TRY_ASSIGN(implementation->embedding_bf16_pipeline,
                      make_compute_pipeline(device, library, @"embedding_bf16"));
        CL_TRY_ASSIGN(implementation->rms_norm_bf16_pipeline,
                      make_compute_pipeline(device, library, @"rms_norm_bf16"));

        CL_TRY_ASSIGN(implementation->gather_rows_f32_pipeline,
                      make_compute_pipeline(device, library, @"gather_rows_f32"));
        CL_TRY_ASSIGN(implementation->linear_bf16_partial_argmax_pipeline,
                      make_compute_pipeline(device, library, @"linear_bf16_partial_argmax"));
        CL_TRY_ASSIGN(implementation->reduce_argmax_pipeline,
                      make_compute_pipeline(device, library, @"reduce_argmax"));

        CL_TRY_ASSIGN(implementation->silu_mul_f32_pipeline,
                      make_compute_pipeline(device, library, @"silu_mul_f32"));
        CL_TRY_ASSIGN(implementation->add_f32_pipeline,
                      make_compute_pipeline(device, library, @"add_f32"));
        CL_TRY_ASSIGN(implementation->rope_f32_pipeline,
                      make_compute_pipeline(device, library, @"rope_f32"));
        CL_TRY_ASSIGN(implementation->store_kv_f32_pipeline,
                      make_compute_pipeline(device, library, @"store_kv_f32"));

        CL_TRY_ASSIGN(implementation->paged_attention_f32_pipeline,
                      make_compute_pipeline(device, library, @"paged_attention_f32"));
        CL_TRY_ASSIGN(implementation->paged_flash_attention_prefill_f32_pipeline,
                      make_compute_pipeline(device, library, @"paged_flash_attention_prefill_f32"));
        CL_TRY_ASSIGN(implementation->paged_attention_partial_f32_pipeline,
                      make_compute_pipeline(device, library, @"paged_attention_partial_f32"));
        CL_TRY_ASSIGN(implementation->paged_attention_reduce_f32_pipeline,
                      make_compute_pipeline(device, library, @"paged_attention_reduce_f32"));

        const char* device_name = device.name.UTF8String;
        implementation->device = device;
        implementation->command_queue = command_queue;
        implementation->shader_library = library;
        implementation->device_name = device_name == nullptr ? "unknown Metal device" : device_name;
        implementation->profiling_enabled = environment_flag("CHIBILLM_PROFILE");
        implementation->tensorops_enabled = implementation->linear_bf16_tensorops_pipeline != nil
            && !environment_flag("CHIBILLM_DISABLE_TENSOROPS");
        implementation->flash_attention_enabled =
            !environment_flag("CHIBILLM_DISABLE_FLASH_ATTENTION");

        std::fprintf(stderr,
                     "[metal] device=%s shaders=%s tensorops=%s flash-attention=%s profile=%s\n",
                     implementation->device_name.c_str(), compiled_metal4 ? "Metal 4" : "legacy",
                     implementation->tensorops_enabled ? "enabled" : "disabled",
                     implementation->flash_attention_enabled ? "enabled" : "disabled",
                     implementation->profiling_enabled ? "detailed" : "off");

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
    if (implementation_ && implementation_->profiling_enabled) {
        implementation_->dump_profile();
        implementation_->arena->dump_stats();
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

        implementation_->arena->begin_pass();
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
        implementation_->arena->complete_pass();

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
        implementation_->arena->complete_pass();
    }
}

result<metal_buffer, metal_error>
metal_context::make_shared_buffer(std::size_t size_bytes) const
{
    @autoreleasepool {
        if (size_bytes == 0) {
            return fail(make_error(metal_errc::invalid_input, "Metal buffer size must be nonzero"));
        }

        const bool is_activation = implementation_->pass_encoder != nil;
        id<MTLBuffer> buffer = is_activation ? implementation_->arena->acquire(size_bytes) : nil;
        if (buffer == nil) {
            buffer = [implementation_->device newBufferWithLength:size_bytes
                                                          options:MTLResourceStorageModeShared];
        }
        if (buffer == nil) {
            return fail(make_error(metal_errc::buffer_creation_failed,
                                   "failed to create a shared Metal buffer"));
        }

        auto buffer_implementation = std::make_unique<metal_buffer::implementation>();
        buffer_implementation->buffer = buffer;
        if (is_activation) {
            buffer_implementation->arena = implementation_->arena;
        }
        return metal_buffer { std::move(buffer_implementation) };
    }
}

} // namespace chibillm
