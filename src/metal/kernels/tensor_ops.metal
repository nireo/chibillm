#include "common.metalh"

template <typename Matrix>
static inline void
embedding_impl(device const int* token_ids,
               Matrix weight,
               device float* output,
               constant uint& token_count,
               constant uint& hidden_size,
               uint2 position)
{
    if (position.x >= hidden_size || position.y >= token_count) {
        return;
    }

    const ulong token_position = position.y;
    const ulong hidden_feature = position.x;
    const ulong hidden_count = hidden_size;
    const ulong token = ulong(token_ids[token_position]);

    const float value = weight.load(token, uint(hidden_feature));

    const ulong output_index = token_position * hidden_count + hidden_feature;
    output[output_index] = value;
}

kernel void
embedding_bf16(device const int* token_ids [[buffer(0)]],
               device const bf16_storage* weight [[buffer(1)]],
               device float* output [[buffer(2)]],
               constant uint& token_count [[buffer(3)]],
               constant uint& hidden_size [[buffer(4)]],
               uint2 position [[thread_position_in_grid]])
{
    embedding_impl(token_ids, bf16_matrix { weight, hidden_size }, output, token_count, hidden_size,
                   position);
}

kernel void
embedding_q4(device const int* token_ids [[buffer(0)]],
             device const uint* weight [[buffer(1)]],
             device float* output [[buffer(2)]],
             constant uint& token_count [[buffer(3)]],
             constant uint& hidden_size [[buffer(4)]],
             device const bf16_storage* scales [[buffer(5)]],
             device const bf16_storage* offsets [[buffer(6)]],
             uint2 position [[thread_position_in_grid]])
{
    embedding_impl(token_ids, q4_matrix { weight, scales, offsets, hidden_size }, output,
                   token_count, hidden_size, position);
}

kernel void
rms_norm_bf16(device const float* input [[buffer(0)]],
              device const bf16_storage* weight [[buffer(1)]],
              device float* output [[buffer(2)]],
              constant uint& row_count [[buffer(3)]],
              constant uint& hidden_size [[buffer(4)]],
              constant float& epsilon [[buffer(5)]],
              constant float& weight_offset [[buffer(6)]],
              uint thread_index [[thread_index_in_threadgroup]],
              uint lane [[thread_index_in_simdgroup]],
              uint simdgroup [[simdgroup_index_in_threadgroup]],
              uint simd_width [[threads_per_simdgroup]],
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

    local_sum = simd_sum(local_sum);
    threadgroup float partial_sums[32];
    if (lane == 0) {
        partial_sums[simdgroup] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        const uint simdgroup_count = thread_count / simd_width;
        float sum = lane < simdgroup_count ? partial_sums[lane] : 0.0F;
        sum = simd_sum(sum);
        if (lane == 0) {
            const float mean_square = sum / float(hidden_size);
            partial_sums[0] = rsqrt(mean_square + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse_rms = partial_sums[0];
    for (uint feature = thread_index; feature < hidden_size; feature += thread_count) {
        const ulong index = row_offset + ulong(feature);
        const float weight_value = load_bf16(weight[feature]) + weight_offset;
        output[index] = input[index] * inverse_rms * weight_value;
    }
}

kernel void
silu_mul_f32(device const float* gate [[buffer(0)]],
             device const float* up [[buffer(1)]],
             device float* output [[buffer(2)]],
             constant uint& element_count [[buffer(3)]],
             constant uint& sigmoid_only [[buffer(4)]],
             uint position [[thread_position_in_grid]])
{
    if (position >= element_count) {
        return;
    }

    const float x = gate[position];
    const float silu = (sigmoid_only ? 1.0F : x) / (1.0F + exp(-x));

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
split_heads_f32(device const float* input [[buffer(0)]],
                device float* first [[buffer(1)]],
                device float* second [[buffer(2)]],
                constant uint& width [[buffer(3)]],
                constant uint& head_dimension [[buffer(4)]],
                uint2 position [[thread_position_in_grid]])
{
    const ulong column = position.x;
    const ulong source = ulong(position.y) * width * 2
        + (column / head_dimension) * head_dimension * 2
        + column % head_dimension;
    const ulong destination = ulong(position.y) * width + column;
    first[destination] = input[source];
    second[destination] = input[source + head_dimension];
}

kernel void
rope_f32(device const float* input [[buffer(0)]],
         device const uint* positions [[buffer(1)]],
         device float* output [[buffer(2)]],
         constant uint& row_count [[buffer(3)]],
         constant uint& head_count [[buffer(4)]],
         constant uint& head_dimension [[buffer(5)]],
         device const float* inverse_frequencies [[buffer(6)]],
         constant uint& rotary_dimension [[buffer(7)]],
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
    const ulong head_offset =
        (ulong(row) * ulong(head_count) + ulong(head)) * ulong(head_dimension);
    const uint half_rotary = rotary_dimension / 2;
    if (pair >= half_rotary) {
        const ulong tail = head_offset + rotary_dimension + 2 * (pair - half_rotary);
        output[tail] = input[tail];
        output[tail + 1] = input[tail + 1];
        return;
    }

    const ulong first_index = head_offset + ulong(pair);
    const ulong second_index = first_index + ulong(half_rotary);

    const float angle = float(positions[row]) * inverse_frequencies[pair];
    const float cosine = cos(angle);
    const float sine = sin(angle);

    const float first = input[first_index];
    const float second = input[second_index];
    output[first_index] = first * cosine - second * sine;
    output[second_index] = second * cosine + first * sine;
}
