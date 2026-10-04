#include "common.metalh"

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

template <typename Matrix>
static inline void
linear_partial_argmax_impl(device const float* input,
                           Matrix weight,
                           device argmax_pair* partials,
                           constant uint& hidden_size,
                           constant uint& vocabulary_size,
                           constant uint& partial_count,
                           constant uint& outputs_per_simdgroup,
                           constant uint& simd_width,
                           uint thread_index,
                           uint lane,
                           uint simdgroup,
                           uint3 threadgroup_position,
                           uint simdgroups_per_threadgroup,
                           threadgroup float* simd_scores,
                           threadgroup uint* simd_indices)
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
            accumulator = weight.dot(input + input_base, output_feature, lane, simd_width);
            if (lane == 0
                && (accumulator > best_score
                    || (accumulator == best_score && output_feature < best_index))) {
                best_score = accumulator;
                best_index = output_feature;
            }
        }
    }

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
    threadgroup float simd_scores[32];
    threadgroup uint simd_indices[32];
    linear_partial_argmax_impl(input, bf16_matrix { weight, hidden_size }, partials, hidden_size,
                               vocabulary_size, partial_count, outputs_per_simdgroup, simd_width,
                               thread_index, lane, simdgroup, threadgroup_position,
                               simdgroups_per_threadgroup, simd_scores, simd_indices);
}

kernel void
linear_affine_partial_argmax(device const float* input [[buffer(0)]],
                             device const uint* weight [[buffer(1)]],
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
                             device const bf16_storage* scales [[buffer(8)]],
                             device const bf16_storage* offsets [[buffer(9)]],
                             uint simdgroups_per_threadgroup [[simdgroups_per_threadgroup]])
{
    threadgroup float simd_scores[32];
    threadgroup uint simd_indices[32];
    linear_partial_argmax_impl(
        input, affine_matrix { weight, scales, offsets, hidden_size }, partials, hidden_size,
        vocabulary_size, partial_count, outputs_per_simdgroup, simd_width, thread_index, lane,
        simdgroup, threadgroup_position, simdgroups_per_threadgroup, simd_scores, simd_indices);
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
