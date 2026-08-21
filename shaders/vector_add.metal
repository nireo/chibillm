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
