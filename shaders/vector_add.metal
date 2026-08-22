#include <metal_stdlib>

using namespace metal;

kernel void
vector_add(device const float* lhs [[buffer(0)]],
           device const float* rhs [[buffer(1)]],
           device float* output [[buffer(2)]],
           constant uint& element_count [[buffer(3)]],
           uint index [[thread_position_in_grid]])
{
    if (index >= element_count) {
        return;
    }

    output[index] = lhs[index] + rhs[index];
}

kernel void
matmul_f32(device const float* lhs [[buffer(0)]],
           device const float* rhs [[buffer(1)]],
           device float* output [[buffer(2)]],
           constant uint& rows [[buffer(3)]],
           constant uint& inner_dimension [[buffer(4)]],
           constant uint& columns [[buffer(5)]],
           uint2 position [[thread_position_in_grid]])
{
    if (position.x >= columns || position.y >= rows) {
        return;
    }

    const ulong row = position.y;
    const ulong column = position.x;
    const ulong k_count = inner_dimension;
    const ulong column_count = columns;

    float accumulator = 0.0F;

    for (ulong k = 0; k < k_count; ++k) {
        const ulong lhs_index = row * k_count + k;
        const ulong rhs_index = k * column_count + column;

        accumulator += lhs[lhs_index] * rhs[rhs_index];
    }

    output[row * column_count + column] = accumulator;
}
