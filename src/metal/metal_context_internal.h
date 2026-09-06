#pragma once
#include "metal/metal_context.h"
#include <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace chibillm {
inline metal_error
make_error(metal_errc code, std::string message)
{
    return { code, std::move(message) };
}

// Exact-size freelists for temporary buffers created while a compute pass is
// open. Released buffers stay pending until the command buffer completes, so a
// later pass can reuse them without aliasing resources inside one encoder.
// Keeping the state shared also makes late buffer destruction safe if its owning
// context has already moved or been destroyed.
struct activation_arena {
    void
    begin_pass()
    {
        const std::scoped_lock lock(mutex);
        pass_open = true;
    }

    void
    complete_pass()
    {
        const std::scoped_lock lock(mutex);
        pass_open = false;
        for (auto& [size_bytes, buffers] : pending) {
            auto& destination = available[size_bytes];
            destination.insert(destination.end(), buffers.begin(), buffers.end());
        }
        pending.clear();
    }

    id<MTLBuffer>
    acquire(std::size_t size_bytes)
    {
        const std::scoped_lock lock(mutex);
        auto found = available.find(size_bytes);
        if (found == available.end() || found->second.empty()) {
            ++allocations;
            return nil;
        }

        id<MTLBuffer> buffer = found->second.back();
        found->second.pop_back();
        cached_bytes -= size_bytes;
        ++reuses;
        return buffer;
    }

    void
    recycle(id<MTLBuffer> buffer)
    {
        const std::scoped_lock lock(mutex);
        const auto size_bytes = static_cast<std::size_t>(buffer.length);
        auto& destination = pass_open ? pending[size_bytes] : available[size_bytes];
        destination.push_back(buffer);
        cached_bytes += size_bytes;
        peak_cached_bytes = std::max(peak_cached_bytes, cached_bytes);
    }

    void
    dump_stats() const
    {
        const std::scoped_lock lock(mutex);
        const auto requests = allocations + reuses;
        const auto reuse_rate = requests == 0 ? 0.0 : 100.0 * double(reuses) / double(requests);
        std::fprintf(stderr,
                     "[metal-arena] allocations=%zu reuses=%zu cached_bytes=%zu "
                     "peak_cached_bytes=%zu reuse_rate=%.1f%%\n",
                     allocations, reuses, cached_bytes, peak_cached_bytes, reuse_rate);
    }

    mutable std::mutex mutex;
    std::map<std::size_t, std::vector<id<MTLBuffer>>> available;
    std::map<std::size_t, std::vector<id<MTLBuffer>>> pending;
    bool pass_open = false;
    std::size_t allocations = 0;
    std::size_t reuses = 0;
    std::size_t cached_bytes = 0;
    std::size_t peak_cached_bytes = 0;
};

struct metal_buffer::implementation {
    id<MTLBuffer> buffer;
    std::shared_ptr<activation_arena> arena;

    ~implementation()
    {
        if (arena != nullptr && buffer != nil) {
            arena->recycle(buffer);
        }
    }
};

struct metal_context::implementation {
    struct profile_stats {
        std::size_t calls = 0;
        double gpu_seconds = 0.0;
    };

    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> shader_library;
    id<MTLComputePipelineState> linear_add_bf16_decode_pipeline;
    id<MTLComputePipelineState> linear_bf16_tensorops_pipeline;
    id<MTLComputePipelineState> linear_split_bf16_pipeline;
    id<MTLComputePipelineState> linear_split_bf16_decode_pipeline;
    id<MTLComputePipelineState> embedding_bf16_pipeline;
    id<MTLComputePipelineState> rms_norm_bf16_pipeline;
    id<MTLComputePipelineState> causal_conv1d_silu_pipeline;
    id<MTLComputePipelineState> gated_delta_rule_pipeline;
    id<MTLComputePipelineState> rms_norm_gated_pipeline;
    id<MTLComputePipelineState> gather_rows_f32_pipeline;
    id<MTLComputePipelineState> linear_bf16_partial_argmax_pipeline;
    id<MTLComputePipelineState> reduce_argmax_pipeline;
    id<MTLComputePipelineState> silu_mul_f32_pipeline;
    id<MTLComputePipelineState> add_f32_pipeline;
    id<MTLComputePipelineState> rope_f32_pipeline;
    id<MTLComputePipelineState> store_kv_f32_pipeline;
    id<MTLComputePipelineState> paged_attention_f32_pipeline;
    id<MTLComputePipelineState> paged_flash_attention_prefill_f32_pipeline;
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
    bool flash_attention_enabled = false;
    std::map<std::string, profile_stats> profile;
    std::mutex rope_frequency_mutex;
    std::map<std::pair<std::uint32_t, std::uint32_t>, id<MTLBuffer>> rope_frequency_buffers;
    std::shared_ptr<activation_arena> arena = std::make_shared<activation_arena>();

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

} // namespace chibillm
