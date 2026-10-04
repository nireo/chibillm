#pragma once
#include "metal/metal_context.h"
#include <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
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
    id<MTLLibrary> kernel_library;
    id<MTLComputePipelineState> linear_add_bf16_decode_pipeline;
    id<MTLComputePipelineState> linear_bf16_tensorops_pipeline;
    id<MTLComputePipelineState> linear_split_bf16_pipeline;
    id<MTLComputePipelineState> linear_split_bf16_decode_pipeline;
    id<MTLComputePipelineState> embedding_bf16_pipeline;

    struct affine_pipelines {
        id<MTLComputePipelineState> embedding;
        id<MTLComputePipelineState> linear_add;
        id<MTLComputePipelineState> linear_split;
        id<MTLComputePipelineState> partial_argmax;
        id<MTLComputePipelineState> expand;
    };

    std::array<affine_pipelines, 2> affine; // Q4, Q8
    // GPU-only writes; reused in dispatch order, independently of activation recycling.
    std::optional<metal_buffer> projection_scratch;
    id<MTLComputePipelineState> rms_norm_bf16_pipeline;
    id<MTLComputePipelineState> causal_conv1d_silu_pipeline;
    id<MTLComputePipelineState> gated_delta_rule_pipeline;
    id<MTLComputePipelineState> delta_prepare_pipeline;
    id<MTLComputePipelineState> delta_products_pipeline;
    id<MTLComputePipelineState> delta_project_pipeline;
    id<MTLComputePipelineState> delta_solve_pipeline;
    id<MTLComputePipelineState> delta_finish_pipeline;

    id<MTLComputePipelineState> rms_norm_gated_pipeline;
    id<MTLComputePipelineState> gather_rows_f32_pipeline;
    id<MTLComputePipelineState> linear_bf16_partial_argmax_pipeline;
    id<MTLComputePipelineState> reduce_argmax_pipeline;
    id<MTLComputePipelineState> silu_mul_f32_pipeline;
    id<MTLComputePipelineState> add_f32_pipeline;
    id<MTLComputePipelineState> rope_f32_pipeline;
    id<MTLComputePipelineState> split_heads_f32_pipeline;
    id<MTLComputePipelineState> store_kv_f32_pipeline;
    id<MTLComputePipelineState> paged_attention_f32_pipeline;
    id<MTLComputePipelineState> paged_attention_simd_f32_pipeline;
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
    bool pass_profiling_enabled = false;
    std::chrono::steady_clock::time_point pass_started;
    bool tensorops_enabled = false;
    bool flash_attention_enabled = false;
    bool simd_attention_enabled = true;
    bool chunkwise_delta_enabled = true;
    std::map<std::string, profile_stats> profile;
    std::mutex rope_frequency_mutex;
    std::map<std::pair<std::uint32_t, std::uint32_t>, id<MTLBuffer>> rope_frequency_buffers;
    std::shared_ptr<activation_arena> arena = std::make_shared<activation_arena>();

    // what one dispatch encodes into; see definition below.
    struct dispatch_frame;

    [[nodiscard]] result<dispatch_frame, metal_error> make_dispatch_encoder();
    [[nodiscard]] id<MTLCommandBuffer> drain_compute_pass();

    template <typename Encode>
    [[nodiscard]] result<void, metal_error> dispatch(std::string_view profile_name, Encode encode);

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

template <typename Encode>
result<void, metal_error>
metal_context::implementation::dispatch(std::string_view profile_name, Encode encode)
{
    auto frame = open_dispatch_encoder();
    if (!frame)
        return fail(frame.error());
    encode(frame->encoder);
    return complete_dispatch_encoder(*frame, profile_name);
}

} // namespace chibillm

namespace chibillm::metal_dispatch {

// Kernel dimensions are uint32_t; callers keep geometry-specific checks nearby.
inline bool
kernel_dimensions_fit(std::initializer_list<std::size_t> dimensions, std::size_t minimum = 0)
{
    return std::ranges::all_of(dimensions, [minimum](const auto dimension) {
        return dimension >= minimum && dimension <= std::numeric_limits<std::uint32_t>::max();
    });
}

template <typename... Buffers>
void
bind_buffers(id<MTLComputeCommandEncoder> encoder,
             NSUInteger first_index,
             const Buffers&... buffers)
{
    ([encoder setBuffer:buffers offset:0 atIndex:first_index++], ...);
}

template <typename... Values>
void
bind_constants(id<MTLComputeCommandEncoder> encoder,
               NSUInteger first_index,
               const Values&... values)
{
    static_assert((std::is_trivially_copyable_v<Values> && ...));
    ([encoder setBytes:&values length:sizeof(values) atIndex:first_index++], ...);
}

inline MTLSize
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

} // namespace chibillm::metal_dispatch
