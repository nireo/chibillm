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

kernel void
rms_norm_bf16(device const float* input [[buffer(0)]],
    device const ushort* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]])
{
    const uint row = threadgroup_position.x;
    if (row >= row_count) {
        return;
    }

    const ulong row_offset = ulong(row) * ulong(hidden_size);
    const uint thread_count = threads_per_threadgroup.x;

    float local_sum = 0.0F;
    for (uint feature = thread_index; feature < hidden_size; feature += thread_count) {
        const float value = input[row_offset + ulong(feature)];
        local_sum += value * value;
    }

    threadgroup float partial_sums[256];
    partial_sums[thread_index] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = thread_count / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            partial_sums[thread_index] += partial_sums[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        const float mean_square = partial_sums[0] / float(hidden_size);
        partial_sums[0] = rsqrt(mean_square + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse_rms = partial_sums[0];
    for (uint feature = thread_index; feature < hidden_size; feature += thread_count) {
        const ulong index = row_offset + ulong(feature);
        const uint weight_bits = uint(weight[feature]) << 16;
        const float weight_value = as_type<float>(weight_bits);
        output[index] = input[index] * inverse_rms * weight_value;
    }
}

kernel void
silu_mul_f32(device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    uint position [[thread_position_in_grid]])
{
    if (position >= element_count) {
        return;
    }

    const float x = gate[position];
    const float silu = x / (1.0F + exp(-x));

    output[position] = silu * up[position];
}

kernel void
add_f32(device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    uint position [[thread_position_in_grid]])
{
    if (position >= element_count) {
        return;
    }

    output[position] = lhs[position] + rhs[position];
}

kernel void
rope_f32(device const float* input [[buffer(0)]],
    device const uint* positions [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& head_count [[buffer(4)]],
    constant uint& head_dimension [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    uint2 grid_position [[thread_position_in_grid]])
{
    const uint half_dimension = head_dimension / 2;
    const uint pair_columns = head_count * half_dimension;
    if (grid_position.x >= pair_columns || grid_position.y >= row_count) {
        return;
    }

    const uint row = grid_position.y;
    const uint head = grid_position.x / half_dimension;
    const uint pair = grid_position.x % half_dimension;
    const ulong head_offset = (ulong(row) * ulong(head_count) + ulong(head)) * ulong(head_dimension);
    const ulong first_index = head_offset + ulong(pair);
    const ulong second_index = first_index + ulong(half_dimension);

    const float exponent = -2.0F * float(pair) / float(head_dimension);
    const float frequency = pow(theta, exponent);
    const float angle = float(positions[row]) * frequency;
    const float cosine = cos(angle);
    const float sine = sin(angle);

    const float first = input[first_index];
    const float second = input[second_index];
    output[first_index] = first * cosine - second * sine;
    output[second_index] = second * cosine + first * sine;
}

kernel void
store_kv_f32(device const float* keys [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const uint* slot_mapping [[buffer(2)]],
    device float* key_cache [[buffer(3)]],
    device float* value_cache [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& feature_count [[buffer(6)]],
    constant uint& layer [[buffer(7)]],
    constant uint& slot_count [[buffer(8)]],
    uint2 position [[thread_position_in_grid]])
{
    if (position.x >= feature_count || position.y >= row_count) {
        return;
    }

    const uint slot = slot_mapping[position.y];
    if (slot >= slot_count) {
        return;
    }

    const ulong row = position.y;
    const ulong feature = position.x;

    const ulong input_index = row * ulong(feature_count) + feature;
    const ulong cache_index = (ulong(layer) * ulong(slot_count) + ulong(slot))
            * ulong(feature_count)
        + feature;

    key_cache[cache_index] = keys[input_index];
    value_cache[cache_index] = values[input_index];
}
