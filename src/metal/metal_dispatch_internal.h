#pragma once

#include "metal/metal_context_internal.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>

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
