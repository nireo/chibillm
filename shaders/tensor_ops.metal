#include <metal_stdlib>

#if defined(CHIBILLM_ENABLE_TENSOROPS) && __METAL_VERSION__ >= 400 && defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#if __METAL_VERSION__ >= 310
using bf16_storage = bfloat;
using bf16x4_storage = bfloat4;

static inline float
load_bf16(bf16_storage value)
{
    return float(value);
}

static inline float4
load_bf16x4(bf16x4_storage value)
{
    return float4(value);
}
#else
using bf16_storage = ushort;
using bf16x4_storage = ushort4;

static inline float
load_bf16(bf16_storage value)
{
    return as_type<float>(uint(value) << 16);
}

static inline float4
load_bf16x4(bf16x4_storage value)
{
    return as_type<float4>(uint4(value) << 16);
}
#endif

#if defined(CHIBILLM_ENABLE_TENSOROPS) && __METAL_VERSION__ >= 400 && defined(__HAVE_TENSOR__)
using namespace mpp::tensor_ops;

// Metal 4 TensorOps kernel for A [rows, input_features] multiplied by the
// transpose of row-major BF16 weights [output_features, input_features]. The
// inline tensor views keep the existing MTLBuffers and command submission path;
// Apple10/M5 executes the matmul operation on each GPU core's neural accelerator.
kernel void
linear_bf16_tensorops(device float* input [[buffer(0)]],
                      device bfloat* weight [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      constant uint& rows [[buffer(3)]],
                      constant uint& input_features [[buffer(4)]],
                      constant uint& output_features [[buffer(5)]],
                      uint2 threadgroup_position [[threadgroup_position_in_grid]])
{
    constexpr int tile_size = 64;
    const int output_origin = int(threadgroup_position.x) * tile_size;
    const int row_origin = int(threadgroup_position.y) * tile_size;

    auto input_tensor = tensor(input, dextents<int, 2>(int(input_features), int(rows)));
    auto weight_tensor =
        tensor(weight, dextents<int, 2>(int(input_features), int(output_features)));
    auto output_tensor = tensor(output, dextents<int, 2>(int(output_features), int(rows)));

    // Weight storage is [output, input], hence transpose_right=true.
    constexpr auto descriptor =
        matmul2d_descriptor(tile_size, tile_size, dynamic_length_v<int>, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    auto input_slice = input_tensor.slice(0, row_origin);
    auto weight_slice = weight_tensor.slice(0, output_origin);
    auto output_slice = output_tensor.slice(output_origin, row_origin);
    auto product = operation.get_destination_cooperative_tensor<decltype(input_slice),
                                                                decltype(weight_slice), float>();
    operation.run(input_slice, weight_slice, product);
    product.store(output_slice);
}
#endif

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
            device const bf16_storage* weight [[buffer(1)]],
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

        const float value = load_bf16(weight[weight_index]);

        accumulator += input[input_index] * value;
    }

    output[row * ulong(output_features) + output_feature] = accumulator;
}

static inline float
linear_bf16_decode_dot(device const float* input,
                       device const bf16_storage* weight,
                       uint input_features,
                       uint weight_row,
                       uint lane,
                       uint simd_width)
{
    float accumulator = 0.0F;
    const ulong weight_base = ulong(weight_row) * ulong(input_features);
    for (uint k = lane * 4; k + 3 < input_features; k += 4 * simd_width) {
        const float4 input_values = { input[k], input[k + 1], input[k + 2], input[k + 3] };
        const bf16x4_storage packed_values = { weight[weight_base + ulong(k)],
                                               weight[weight_base + ulong(k + 1)],
                                               weight[weight_base + ulong(k + 2)],
                                               weight[weight_base + ulong(k + 3)] };
        const float4 weight_values = load_bf16x4(packed_values);
        accumulator += dot(input_values, weight_values);
    }
    const uint vectorized_features = input_features & ~3u;
    for (uint k = vectorized_features + lane; k < input_features; k += simd_width) {
        const float value = load_bf16(weight[weight_base + ulong(k)]);
        accumulator += input[k] * value;
    }
    return simd_sum(accumulator);
}

kernel void
linear_bf16_decode(device const float* input [[buffer(0)]],
                   device const bf16_storage* weight [[buffer(1)]],
                   device float* output [[buffer(2)]],
                   constant uint& input_features [[buffer(3)]],
                   constant uint& output_features [[buffer(4)]],
                   constant uint& outputs_per_threadgroup [[buffer(5)]],
                   constant uint& simd_width [[buffer(6)]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint simdgroup [[simdgroup_index_in_threadgroup]],
                   uint3 threadgroup_position [[threadgroup_position_in_grid]])
{
    const uint output_feature = threadgroup_position.x * outputs_per_threadgroup + simdgroup;
    if (output_feature >= output_features) {
        return;
    }

    const float accumulator =
        linear_bf16_decode_dot(input, weight, input_features, output_feature, lane, simd_width);
    if (lane == 0) {
        output[output_feature] = accumulator;
    }
}

kernel void
linear_add_bf16_decode(device const float* input [[buffer(0)]],
                       device const bf16_storage* weight [[buffer(1)]],
                       device const float* residual [[buffer(2)]],
                       device float* output [[buffer(3)]],
                       constant uint& input_features [[buffer(4)]],
                       constant uint& output_features [[buffer(5)]],
                       constant uint& outputs_per_threadgroup [[buffer(6)]],
                       constant uint& simd_width [[buffer(7)]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simdgroup [[simdgroup_index_in_threadgroup]],
                       uint3 threadgroup_position [[threadgroup_position_in_grid]])
{
    const uint output_feature = threadgroup_position.x * outputs_per_threadgroup + simdgroup;
    if (output_feature >= output_features) {
        return;
    }

    const float accumulator =
        linear_bf16_decode_dot(input, weight, input_features, output_feature, lane, simd_width);
    if (lane == 0) {
        output[output_feature] = residual[output_feature] + accumulator;
    }
}

kernel void
linear_split_bf16(device const float* input [[buffer(0)]],
                  device const bf16_storage* weight [[buffer(1)]],
                  device float* output_a [[buffer(2)]],
                  device float* output_b [[buffer(3)]],
                  device float* output_c [[buffer(4)]],
                  constant uint& rows [[buffer(5)]],
                  constant uint& input_features [[buffer(6)]],
                  constant uint& width_a [[buffer(7)]],
                  constant uint& width_b [[buffer(8)]],
                  constant uint& width_c [[buffer(9)]],
                  uint2 position [[thread_position_in_grid]])
{
    const uint total_width = width_a + width_b + width_c;
    if (position.x >= total_width || position.y >= rows) {
        return;
    }

    const ulong row = position.y;
    const ulong weight_column = position.x;
    const ulong k_count = input_features;

    device float* output;
    ulong row_length;
    ulong local_column;
    if (weight_column < width_a) {
        output = output_a;
        row_length = width_a;
        local_column = weight_column;
    } else if (weight_column < width_a + width_b) {
        output = output_b;
        row_length = width_b;
        local_column = weight_column - width_a;
    } else {
        output = output_c;
        row_length = width_c;
        local_column = weight_column - width_a - width_b;
    }

    float accumulator = 0.0F;
    const ulong weight_base = weight_column * k_count;
    for (ulong k = 0; k < k_count; ++k) {
        const float value = load_bf16(weight[weight_base + k]);
        accumulator += input[row * k_count + k] * value;
    }

    output[row * row_length + local_column] = accumulator;
}

kernel void
linear_split_bf16_decode(device const float* input [[buffer(0)]],
                         device const bf16_storage* weight [[buffer(1)]],
                         device float* output_a [[buffer(2)]],
                         device float* output_b [[buffer(3)]],
                         device float* output_c [[buffer(4)]],
                         constant uint& input_features [[buffer(5)]],
                         constant uint& width_a [[buffer(6)]],
                         constant uint& width_b [[buffer(7)]],
                         constant uint& width_c [[buffer(8)]],
                         constant uint& outputs_per_threadgroup [[buffer(9)]],
                         constant uint& simd_width [[buffer(10)]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simdgroup [[simdgroup_index_in_threadgroup]],
                         uint3 threadgroup_position [[threadgroup_position_in_grid]])
{
    const uint weight_row = threadgroup_position.x * outputs_per_threadgroup + simdgroup;
    const uint total_width = width_a + width_b + width_c;
    if (weight_row >= total_width) {
        return;
    }

    const float accumulator =
        linear_bf16_decode_dot(input, weight, input_features, weight_row, lane, simd_width);
    if (lane != 0) {
        return;
    }
    if (weight_row < width_a) {
        output_a[weight_row] = accumulator;
    } else if (weight_row < width_a + width_b) {
        output_b[weight_row - width_a] = accumulator;
    } else {
        output_c[weight_row - width_a - width_b] = accumulator;
    }
}

kernel void
embedding_bf16(device const int* token_ids [[buffer(0)]],
               device const bf16_storage* weight [[buffer(1)]],
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
    const float value = load_bf16(weight[weight_index]);

    const ulong output_index = token_position * hidden_count + hidden_feature;
    output[output_index] = value;
}

kernel void
rms_norm_bf16(device const float* input [[buffer(0)]],
              device const bf16_storage* weight [[buffer(1)]],
              device float* output [[buffer(2)]],
              constant uint& row_count [[buffer(3)]],
              constant uint& hidden_size [[buffer(4)]],
              constant float& epsilon [[buffer(5)]],
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
        const float weight_value = load_bf16(weight[feature]);
        output[index] = input[index] * inverse_rms * weight_value;
    }
}

// Output sampling keeps the selected rows and vocabulary scores on the GPU.
// Each projection threadgroup emits only its best score/token pair; a second
// small reduction turns those partials into one token ID per requested row.
struct argmax_pair {
    float score;
    uint index;
};

kernel void
gather_rows_f32(device const float* input [[buffer(0)]],
                device const uint* row_indices [[buffer(1)]],
                device float* output [[buffer(2)]],
                constant uint& hidden_size [[buffer(3)]],
                uint2 position [[thread_position_in_grid]])
{
    if (position.x < hidden_size) {
        output[ulong(position.y) * hidden_size + position.x] =
            input[ulong(row_indices[position.y]) * hidden_size + position.x];
    }
}

kernel void
linear_bf16_partial_argmax(device const float* input [[buffer(0)]],
                           device const bf16_storage* weight [[buffer(1)]],
                           device argmax_pair* partials [[buffer(2)]],
                           constant uint& hidden_size [[buffer(3)]],
                           constant uint& vocabulary_size [[buffer(4)]],
                           constant uint& partial_count [[buffer(5)]],
                           constant uint& outputs_per_simdgroup [[buffer(6)]],
                           constant uint& simd_width [[buffer(7)]],
                           uint thread_index [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simdgroup [[simdgroup_index_in_threadgroup]],
                           uint3 threadgroup_position [[threadgroup_position_in_grid]],
                           uint simdgroups_per_threadgroup [[simdgroups_per_threadgroup]])
{
    const uint row = threadgroup_position.y;
    const uint outputs_per_threadgroup = simdgroups_per_threadgroup * outputs_per_simdgroup;
    const uint first_output =
        threadgroup_position.x * outputs_per_threadgroup + simdgroup * outputs_per_simdgroup;
    float best_score = -INFINITY;
    uint best_index = 0xffffffffu;

    for (uint local_output = 0; local_output < outputs_per_simdgroup; ++local_output) {
        const uint output_feature = first_output + local_output;
        float accumulator = 0.0F;
        if (output_feature < vocabulary_size) {
            const ulong input_base = ulong(row) * ulong(hidden_size);
            accumulator = linear_bf16_decode_dot(input + input_base, weight, hidden_size,
                                                 output_feature, lane, simd_width);
            if (lane == 0
                && (accumulator > best_score
                    || (accumulator == best_score && output_feature < best_index))) {
                best_score = accumulator;
                best_index = output_feature;
            }
        }
    }

    threadgroup float simd_scores[32];
    threadgroup uint simd_indices[32];
    if (lane == 0) {
        simd_scores[simdgroup] = best_score;
        simd_indices[simdgroup] = best_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        for (uint group = 1; group < simdgroups_per_threadgroup; ++group) {
            if (simd_scores[group] > simd_scores[0]
                || (simd_scores[group] == simd_scores[0]
                    && simd_indices[group] < simd_indices[0])) {
                simd_scores[0] = simd_scores[group];
                simd_indices[0] = simd_indices[group];
            }
        }
        partials[ulong(row) * ulong(partial_count) + threadgroup_position.x] = { simd_scores[0],
                                                                                 simd_indices[0] };
    }
}

kernel void
reduce_argmax(device const argmax_pair* partials [[buffer(0)]],
              device uint* token_ids [[buffer(1)]],
              constant uint& partial_count [[buffer(2)]],
              uint thread_index [[thread_index_in_threadgroup]],
              uint3 threadgroup_position [[threadgroup_position_in_grid]],
              uint3 threads_per_threadgroup [[threads_per_threadgroup]])
{
    const uint row = threadgroup_position.x;
    float best_score = -INFINITY;
    uint best_index = 0xffffffffu;
    for (uint partial = thread_index; partial < partial_count;
         partial += threads_per_threadgroup.x) {
        const argmax_pair candidate = partials[ulong(row) * ulong(partial_count) + partial];
        if (candidate.score > best_score
            || (candidate.score == best_score && candidate.index < best_index)) {
            best_score = candidate.score;
            best_index = candidate.index;
        }
    }

    threadgroup float scores[256];
    threadgroup uint indices[256];
    scores[thread_index] = best_score;
    indices[thread_index] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_threadgroup.x / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride
            && (scores[thread_index + stride] > scores[thread_index]
                || (scores[thread_index + stride] == scores[thread_index]
                    && indices[thread_index + stride] < indices[thread_index]))) {
            scores[thread_index] = scores[thread_index + stride];
            indices[thread_index] = indices[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0) {
        token_ids[row] = indices[0];
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
         device const float* inverse_frequencies [[buffer(6)]],
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
    const ulong first_index = head_offset + ulong(pair);
    const ulong second_index = first_index + ulong(half_dimension);

    const float angle = float(positions[row]) * inverse_frequencies[pair];
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
    const ulong cache_index =
        (ulong(layer) * ulong(slot_count) + ulong(slot)) * ulong(feature_count) + feature;

    key_cache[cache_index] = keys[input_index];
    value_cache[cache_index] = values[input_index];
}

kernel void
paged_attention_f32(device const float* queries [[buffer(0)]],
                    device const uint* positions [[buffer(1)]],
                    device const uint* block_table [[buffer(2)]],
                    device const uint* block_table_offsets [[buffer(3)]],
                    device const uint* block_table_lengths [[buffer(4)]],
                    device const float* key_cache [[buffer(5)]],
                    device const float* value_cache [[buffer(6)]],
                    device float* output [[buffer(7)]],
                    constant uint& row_count [[buffer(8)]],
                    constant uint& query_head_count [[buffer(9)]],
                    constant uint& kv_head_count [[buffer(10)]],
                    constant uint& head_dimension [[buffer(11)]],
                    constant uint& block_size [[buffer(12)]],
                    constant uint& slot_count [[buffer(13)]],
                    constant uint& layer [[buffer(14)]],
                    constant uint& block_table_entry_count [[buffer(15)]],
                    constant uint& simdgroup_count [[buffer(16)]],
                    threadgroup float* score_scratch [[threadgroup(0)]],
                    threadgroup float* softmax_state [[threadgroup(1)]],
                    uint thread_index [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint simdgroup [[simdgroup_index_in_threadgroup]],
                    uint3 threadgroup_position [[threadgroup_position_in_grid]])
{
    // one threadgroup owns one [query row, query head]. Each thread owns one
    // feature of that head and eventually writes the matching output feature.
    const uint query_head = threadgroup_position.x;
    const uint row = threadgroup_position.y;
    if (query_head >= query_head_count || row >= row_count || thread_index >= head_dimension) {
        return;
    }

    // locate this row's slice in the batch's flattened block-table tensor.
    // The query may attend to logical token positions [0, query_position].
    const uint query_position = positions[row];
    const uint table_offset = block_table_offsets[row];
    const uint table_length = block_table_lengths[row];
    const uint last_logical_block = query_position / block_size;
    if (table_offset > block_table_entry_count
        || table_length > block_table_entry_count - table_offset
        || last_logical_block >= table_length) {
        return;
    }

    // grouped-query attention lets several query heads share one cached KV head.
    const uint kv_group_size = query_head_count / kv_head_count;
    const uint kv_head = query_head / kv_group_size;
    const ulong query_base =
        (ulong(row) * ulong(query_head_count) + ulong(query_head)) * ulong(head_dimension);
    const float query_feature = queries[query_base + ulong(thread_index)];

    // shared online-softmax state: running max, denominator, old-state rescale,
    // and current-token weight. Only thread 0 mutates it.
    if (thread_index == 0) {
        softmax_state[0] = -INFINITY;
        softmax_state[1] = 0.0F;
        softmax_state[2] = 0.0F;
        softmax_state[3] = 0.0F;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // this accumulator is private: each thread accumulates one value feature.
    float value_accumulator = 0.0F;
    const float attention_scale = rsqrt(float(head_dimension));

    for (uint token_position = 0;; ++token_position) {
        // translate the logical token position through the sequence's block
        // table to find its physical KV-cache slot.
        const uint logical_block = token_position / block_size;
        const uint token_offset = token_position % block_size;
        const uint physical_block = block_table[table_offset + logical_block];
        const uint slot = physical_block * block_size + token_offset;
        // flatten [layer][slot][KV head][feature], all threads share the base;
        // thread_index selects the feature owned by this thread.
        const ulong cache_base =
            ((ulong(layer) * ulong(slot_count) + ulong(slot)) * ulong(kv_head_count)
             + ulong(kv_head))
            * ulong(head_dimension);
        const ulong cache_index = cache_base + ulong(thread_index);

        // Reduce each SIMD group's contiguous slice of q dot k in registers.
        const float partial_score = simd_sum(query_feature * key_cache[cache_index]);
        if (lane == 0) {
            score_scratch[simdgroup] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Only the few SIMD-group partials remain for the shared softmax state.
        if (thread_index == 0) {
            float score = 0.0F;
            for (uint group = 0; group < simdgroup_count; ++group) {
                score += score_scratch[group];
            }
            score *= attention_scale;

            // re-express old weights relative to the new running maximum, then
            // add this token without storing all preceding attention scores.
            const float old_max = softmax_state[0];
            const float new_max = max(old_max, score);
            const float accumulator_rescale = exp(old_max - new_max);
            const float value_weight = exp(score - new_max);

            softmax_state[0] = new_max;
            softmax_state[1] = softmax_state[1] * accumulator_rescale + value_weight;
            softmax_state[2] = accumulator_rescale;
            softmax_state[3] = value_weight;
        }
        // publish the new rescale and token weight to every feature thread.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        value_accumulator =
            value_accumulator * softmax_state[2] + value_cache[cache_index] * softmax_state[3];
        // no thread may overwrite shared state for the next token until every
        // feature has consumed the current rescale and weight.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token_position == query_position) {
            break;
        }
    }

    // normalizing the weighted-value numerator produces this output feature.
    output[query_base + ulong(thread_index)] = value_accumulator / softmax_state[1];
}

// Paged FlashAttention-style prefill. One SIMD group owns one query-head tile,
// keeps eight query/output rows in registers, and streams sixteen-key tiles
// through a small threadgroup score buffer. Keys and values are therefore read
// once per query tile instead of once per query row, while the online-softmax
// recurrence avoids ever materializing the complete attention matrix.
kernel void
paged_flash_attention_prefill_f32(device const float* queries [[buffer(0)]],
                                  device const uint* positions [[buffer(1)]],
                                  device const uint* block_table [[buffer(2)]],
                                  device const uint* block_table_offsets [[buffer(3)]],
                                  device const uint* block_table_lengths [[buffer(4)]],
                                  device const float* key_cache [[buffer(5)]],
                                  device const float* value_cache [[buffer(6)]],
                                  device const uint* query_tile_starts [[buffer(7)]],
                                  device const uint* query_tile_lengths [[buffer(8)]],
                                  device float* output [[buffer(9)]],
                                  constant uint& row_count [[buffer(10)]],
                                  constant uint& query_head_count [[buffer(11)]],
                                  constant uint& kv_head_count [[buffer(12)]],
                                  constant uint& head_dimension [[buffer(13)]],
                                  constant uint& block_size [[buffer(14)]],
                                  constant uint& slot_count [[buffer(15)]],
                                  constant uint& layer [[buffer(16)]],
                                  constant uint& block_table_entry_count [[buffer(17)]],
                                  constant uint& query_tile_count [[buffer(18)]],
                                  constant uint& simd_width [[buffer(19)]],
                                  threadgroup float* scores [[threadgroup(0)]],
                                  threadgroup float* accumulator_rescales [[threadgroup(1)]],
                                  threadgroup float* normalizers [[threadgroup(2)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint3 threadgroup_position [[threadgroup_position_in_grid]])
{
    constexpr uint query_tile_size = 8;
    constexpr uint key_tile_size = 16;
    constexpr uint features_per_lane = 4;

    const uint query_head = threadgroup_position.x;
    const uint query_tile = threadgroup_position.y;
    if (query_head >= query_head_count || query_tile >= query_tile_count) {
        return;
    }

    const uint row_start = query_tile_starts[query_tile];
    const uint tile_rows = query_tile_lengths[query_tile];
    if (tile_rows == 0
        || tile_rows > query_tile_size
        || row_start >= row_count
        || tile_rows > row_count - row_start
        || head_dimension > simd_width * features_per_lane) {
        return;
    }

    const uint table_offset = block_table_offsets[row_start];
    const uint table_length = block_table_lengths[row_start];
    uint query_positions[query_tile_size];
    ulong query_bases[query_tile_size];
    float query_features[query_tile_size][features_per_lane];
    float output_accumulators[query_tile_size][features_per_lane];

    for (uint query = 0; query < query_tile_size; ++query) {
        const bool active = query < tile_rows;
        const uint row = active ? row_start + query : row_start;
        query_positions[query] = active ? positions[row] : 0;
        query_bases[query] =
            (ulong(row) * ulong(query_head_count) + ulong(query_head)) * ulong(head_dimension);
        for (uint component = 0; component < features_per_lane; ++component) {
            const uint feature = lane + component * simd_width;
            query_features[query][component] = active && feature < head_dimension
                ? queries[query_bases[query] + ulong(feature)]
                : 0.0F;
            output_accumulators[query][component] = 0.0F;
        }
    }

    const uint final_position = query_positions[tile_rows - 1];
    const uint final_logical_block = final_position / block_size;
    if (table_offset > block_table_entry_count
        || table_length > block_table_entry_count - table_offset
        || final_logical_block >= table_length) {
        return;
    }

    const uint kv_group_size = query_head_count / kv_head_count;
    const uint kv_head = query_head / kv_group_size;
    const float attention_scale = rsqrt(float(head_dimension));
    float running_maxima[query_tile_size];
    float running_sums[query_tile_size];
    for (uint query = 0; query < query_tile_size; ++query) {
        running_maxima[query] = -INFINITY;
        running_sums[query] = 0.0F;
    }

    for (uint key_tile_begin = 0; key_tile_begin <= final_position;
         key_tile_begin += key_tile_size) {
        const uint tile_keys = min(key_tile_size, final_position + 1 - key_tile_begin);

        // Compute the BQ x BK score tile. A lane owns up to four head
        // features; the SIMD reduction produces a complete q dot k score.
        for (uint key = 0; key < tile_keys; ++key) {
            const uint token_position = key_tile_begin + key;
            const uint logical_block = token_position / block_size;
            const uint token_offset = token_position % block_size;
            const uint physical_block = block_table[table_offset + logical_block];
            const uint slot = physical_block * block_size + token_offset;
            const ulong cache_base =
                ((ulong(layer) * ulong(slot_count) + ulong(slot)) * ulong(kv_head_count)
                 + ulong(kv_head))
                * ulong(head_dimension);

            float key_features[features_per_lane];
            for (uint component = 0; component < features_per_lane; ++component) {
                const uint feature = lane + component * simd_width;
                key_features[component] =
                    feature < head_dimension ? key_cache[cache_base + ulong(feature)] : 0.0F;
            }

            for (uint query = 0; query < tile_rows; ++query) {
                float partial_score = 0.0F;
                for (uint component = 0; component < features_per_lane; ++component) {
                    partial_score += query_features[query][component] * key_features[component];
                }
                const float score = simd_sum(partial_score) * attention_scale;
                if (lane == 0) {
                    scores[query * key_tile_size + key] =
                        token_position <= query_positions[query] ? score : -INFINITY;
                }
            }
        }

        // Lane zero merges this score tile into each row's stable online
        // softmax state and replaces scores with the value weights referenced
        // to the new running maximum.
        if (lane == 0) {
            for (uint query = 0; query < tile_rows; ++query) {
                float tile_maximum = -INFINITY;
                for (uint key = 0; key < tile_keys; ++key) {
                    tile_maximum = max(tile_maximum, scores[query * key_tile_size + key]);
                }

                const float new_maximum = max(running_maxima[query], tile_maximum);
                const float old_rescale = exp(running_maxima[query] - new_maximum);
                float tile_sum = 0.0F;
                for (uint key = 0; key < tile_keys; ++key) {
                    const float weight = exp(scores[query * key_tile_size + key] - new_maximum);
                    scores[query * key_tile_size + key] = weight;
                    tile_sum += weight;
                }

                running_maxima[query] = new_maximum;
                running_sums[query] = running_sums[query] * old_rescale + tile_sum;
                accumulator_rescales[query] = old_rescale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reuse every loaded value across all query rows.
        for (uint component = 0; component < features_per_lane; ++component) {
            const uint feature = lane + component * simd_width;
            if (feature >= head_dimension) {
                continue;
            }
            for (uint query = 0; query < tile_rows; ++query) {
                output_accumulators[query][component] *= accumulator_rescales[query];
            }
            for (uint key = 0; key < tile_keys; ++key) {
                const uint token_position = key_tile_begin + key;
                const uint logical_block = token_position / block_size;
                const uint token_offset = token_position % block_size;
                const uint physical_block = block_table[table_offset + logical_block];
                const uint slot = physical_block * block_size + token_offset;
                const ulong cache_index =
                    (((ulong(layer) * ulong(slot_count) + ulong(slot)) * ulong(kv_head_count)
                      + ulong(kv_head))
                     * ulong(head_dimension))
                    + ulong(feature);
                const float value = value_cache[cache_index];
                for (uint query = 0; query < tile_rows; ++query) {
                    output_accumulators[query][component] +=
                        scores[query * key_tile_size + key] * value;
                }
            }
        }
        // All lanes must consume the score/weight tile before lane zero
        // overwrites it during the next key tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        for (uint query = 0; query < tile_rows; ++query) {
            normalizers[query] = running_sums[query];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint query = 0; query < tile_rows; ++query) {
        for (uint component = 0; component < features_per_lane; ++component) {
            const uint feature = lane + component * simd_width;
            if (feature < head_dimension) {
                output[query_bases[query] + ulong(feature)] =
                    output_accumulators[query][component] / normalizers[query];
            }
        }
    }
}

kernel void
paged_attention_partial_f32(device const float* queries [[buffer(0)]],
                            device const uint* positions [[buffer(1)]],
                            device const uint* block_table [[buffer(2)]],
                            device const uint* block_table_offsets [[buffer(3)]],
                            device const uint* block_table_lengths [[buffer(4)]],
                            device const float* key_cache [[buffer(5)]],
                            device const float* value_cache [[buffer(6)]],
                            device float* partials [[buffer(7)]],
                            constant uint& row_count [[buffer(8)]],
                            constant uint& query_head_count [[buffer(9)]],
                            constant uint& kv_head_count [[buffer(10)]],
                            constant uint& head_dimension [[buffer(11)]],
                            constant uint& block_size [[buffer(12)]],
                            constant uint& slot_count [[buffer(13)]],
                            constant uint& layer [[buffer(14)]],
                            constant uint& block_table_entry_count [[buffer(15)]],
                            constant uint& simdgroup_count [[buffer(16)]],
                            constant uint& chunk_size [[buffer(17)]],
                            constant uint& chunk_count [[buffer(18)]],
                            threadgroup float* score_scratch [[threadgroup(0)]],
                            threadgroup float* softmax_state [[threadgroup(1)]],
                            uint thread_index [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simdgroup [[simdgroup_index_in_threadgroup]],
                            uint3 threadgroup_position [[threadgroup_position_in_grid]])
{
    const uint query_head = threadgroup_position.x % query_head_count;
    const uint chunk = threadgroup_position.x / query_head_count;
    const uint row = threadgroup_position.y;
    if (chunk >= chunk_count || row >= row_count || thread_index >= head_dimension) {
        return;
    }

    const uint query_position = positions[row];
    const uint table_offset = block_table_offsets[row];
    const uint table_length = block_table_lengths[row];
    const uint last_logical_block = query_position / block_size;
    if (table_offset > block_table_entry_count
        || table_length > block_table_entry_count - table_offset
        || last_logical_block >= table_length) {
        return;
    }

    const uint chunk_begin = chunk * chunk_size;
    const uint chunk_end = min(query_position + 1, chunk_begin + chunk_size);
    const ulong partial_base =
        ((ulong(row) * ulong(query_head_count) + ulong(query_head)) * ulong(chunk_count)
         + ulong(chunk))
        * ulong(head_dimension + 2);
    if (chunk_begin >= chunk_end) {
        if (thread_index == 0) {
            partials[partial_base] = -INFINITY;
            partials[partial_base + 1] = 0.0F;
        }
        partials[partial_base + 2 + ulong(thread_index)] = 0.0F;
        return;
    }

    const uint kv_group_size = query_head_count / kv_head_count;
    const uint kv_head = query_head / kv_group_size;
    const ulong query_base =
        (ulong(row) * ulong(query_head_count) + ulong(query_head)) * ulong(head_dimension);
    const float query_feature = queries[query_base + ulong(thread_index)];

    if (thread_index == 0) {
        softmax_state[0] = -INFINITY;
        softmax_state[1] = 0.0F;
        softmax_state[2] = 0.0F;
        softmax_state[3] = 0.0F;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float value_accumulator = 0.0F;
    const float attention_scale = rsqrt(float(head_dimension));
    for (uint token_position = chunk_begin; token_position < chunk_end; ++token_position) {
        const uint logical_block = token_position / block_size;
        const uint token_offset = token_position % block_size;
        const uint physical_block = block_table[table_offset + logical_block];
        const uint slot = physical_block * block_size + token_offset;
        const ulong cache_base =
            ((ulong(layer) * ulong(slot_count) + ulong(slot)) * ulong(kv_head_count)
             + ulong(kv_head))
            * ulong(head_dimension);
        const ulong cache_index = cache_base + ulong(thread_index);

        const float partial_score = simd_sum(query_feature * key_cache[cache_index]);
        if (lane == 0) {
            score_scratch[simdgroup] = partial_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_index == 0) {
            float score = 0.0F;
            for (uint group = 0; group < simdgroup_count; ++group) {
                score += score_scratch[group];
            }
            score *= attention_scale;

            const float old_max = softmax_state[0];
            const float new_max = max(old_max, score);
            const float accumulator_rescale = exp(old_max - new_max);
            const float value_weight = exp(score - new_max);
            softmax_state[0] = new_max;
            softmax_state[1] = softmax_state[1] * accumulator_rescale + value_weight;
            softmax_state[2] = accumulator_rescale;
            softmax_state[3] = value_weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        value_accumulator =
            value_accumulator * softmax_state[2] + value_cache[cache_index] * softmax_state[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        partials[partial_base] = softmax_state[0];
        partials[partial_base + 1] = softmax_state[1];
    }
    partials[partial_base + 2 + ulong(thread_index)] = value_accumulator;
}

kernel void
paged_attention_reduce_f32(device const float* partials [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& row_count [[buffer(2)]],
                           constant uint& query_head_count [[buffer(3)]],
                           constant uint& head_dimension [[buffer(4)]],
                           constant uint& chunk_count [[buffer(5)]],
                           threadgroup float* chunk_scales [[threadgroup(0)]],
                           threadgroup float* softmax_state [[threadgroup(1)]],
                           uint thread_index [[thread_index_in_threadgroup]],
                           uint3 threadgroup_position [[threadgroup_position_in_grid]])
{
    const uint query_head = threadgroup_position.x;
    const uint row = threadgroup_position.y;
    if (query_head >= query_head_count || row >= row_count || thread_index >= head_dimension) {
        return;
    }

    const ulong head_base = (ulong(row) * ulong(query_head_count) + ulong(query_head))
        * ulong(chunk_count)
        * ulong(head_dimension + 2);
    if (thread_index == 0) {
        float global_max = -INFINITY;
        for (uint chunk = 0; chunk < chunk_count; ++chunk) {
            const ulong base = head_base + ulong(chunk) * ulong(head_dimension + 2);
            global_max = max(global_max, partials[base]);
        }

        float denominator = 0.0F;
        for (uint chunk = 0; chunk < chunk_count; ++chunk) {
            const ulong base = head_base + ulong(chunk) * ulong(head_dimension + 2);
            const float scale = exp(partials[base] - global_max);
            chunk_scales[chunk] = scale;
            denominator += partials[base + 1] * scale;
        }
        softmax_state[0] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulator = 0.0F;
    for (uint chunk = 0; chunk < chunk_count; ++chunk) {
        const ulong base = head_base + ulong(chunk) * ulong(head_dimension + 2);
        accumulator += partials[base + 2 + ulong(thread_index)] * chunk_scales[chunk];
    }
    const ulong output_index =
        (ulong(row) * ulong(query_head_count) + ulong(query_head)) * ulong(head_dimension)
        + ulong(thread_index);
    output[output_index] = accumulator / softmax_state[0];
}
