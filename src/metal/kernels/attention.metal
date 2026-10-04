#include "common.metalh"

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

// One SIMD group owns a query/head. Keeping the online softmax in registers
// avoids cross-SIMD reductions and threadgroup barriers for every cached token.
kernel void
paged_attention_simd_f32(device const float* queries [[buffer(0)]],
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
                         constant uint& simd_width [[buffer(16)]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint3 group [[threadgroup_position_in_grid]])
{
    constexpr uint features_per_lane = 8;
    const uint head = group.x, row = group.y;
    if (head >= query_head_count
        || row >= row_count
        || head_dimension > simd_width * features_per_lane)
        return;
    const uint position = positions[row];
    const uint table_offset = block_table_offsets[row];
    const uint table_length = block_table_lengths[row];
    if (table_offset > block_table_entry_count
        || table_length > block_table_entry_count - table_offset
        || position / block_size >= table_length)
        return;

    const uint kv_head = head / (query_head_count / kv_head_count);
    const ulong query_base = (ulong(row) * query_head_count + head) * head_dimension;
    float query[features_per_lane], accumulator[features_per_lane];
    for (uint c = 0; c < features_per_lane; ++c) {
        const uint feature = lane + c * simd_width;
        query[c] = feature < head_dimension ? queries[query_base + feature] : 0.0F;
        accumulator[c] = 0.0F;
    }
    float maximum = -INFINITY, denominator = 0.0F;
    const float scale = rsqrt(float(head_dimension));
    for (uint token = 0;; ++token) {
        const uint block = block_table[table_offset + token / block_size];
        const uint slot = block * block_size + token % block_size;
        const ulong cache_base =
            ((ulong(layer) * slot_count + slot) * kv_head_count + kv_head) * head_dimension;
        float score = 0.0F;
        for (uint c = 0; c < features_per_lane; ++c) {
            const uint feature = lane + c * simd_width;
            if (feature < head_dimension)
                score += query[c] * key_cache[cache_base + feature];
        }
        score = simd_sum(score) * scale;
        const float new_maximum = max(maximum, score);
        const float rescale = exp(maximum - new_maximum);
        const float weight = exp(score - new_maximum);
        denominator = denominator * rescale + weight;
        maximum = new_maximum;
        for (uint c = 0; c < features_per_lane; ++c) {
            const uint feature = lane + c * simd_width;
            if (feature < head_dimension)
                accumulator[c] =
                    accumulator[c] * rescale + weight * value_cache[cache_base + feature];
        }
        if (token == position)
            break;
    }
    for (uint c = 0; c < features_per_lane; ++c) {
        const uint feature = lane + c * simd_width;
        if (feature < head_dimension)
            output[query_base + feature] = accumulator[c] / denominator;
    }
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
