#include <metal_stdlib>

using namespace metal;

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

kernel void
linear_bf16(device const float* input [[buffer(0)]],
            device const ushort* weight [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& rows [[buffer(3)]],
            constant uint& input_features [[buffer(4)]],
            constant uint& output_features [[buffer(5)]],
            uint2 position [[thread_position_in_grid]])
{
    if (position.x >= output_features || position.y >= rows) {
        return;
    }

    const ulong row = position.y;
    const ulong output_feature = position.x;
    const ulong k_count = input_features;

    float accumulator = 0.0F;
    for (ulong k = 0; k < k_count; ++k) {
        const ulong input_index = row * k_count + k;
        const ulong weight_index = output_feature * k_count + k;

        const uint float_bits = uint(weight[weight_index]) << 16;
        const float value = as_type<float>(float_bits);

        accumulator += input[input_index] * value;
    }

    output[row * ulong(output_features) + output_feature] = accumulator;
}

kernel void
embedding_bf16(device const int* token_ids [[buffer(0)]],
               device const ushort* weight [[buffer(1)]],
               device float* output [[buffer(2)]],
               constant uint& token_count [[buffer(3)]],
               constant uint& hidden_size [[buffer(4)]],
               uint2 position [[thread_position_in_grid]])
{
    if (position.x >= hidden_size || position.y >= token_count) {
        return;
    }

    const ulong token_position = position.y;
    const ulong hidden_feature = position.x;
    const ulong hidden_count = hidden_size;
    const ulong token = ulong(token_ids[token_position]);

    const ulong weight_index = token * hidden_count + hidden_feature;
    const uint float_bits = uint(weight[weight_index]) << 16;
    const float value = as_type<float>(float_bits);

    const ulong output_index = token_position * hidden_count + hidden_feature;
    output[output_index] = value;
}
