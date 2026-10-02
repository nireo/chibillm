#include "qwen/qwen_layer.h"
#include "tensor/layers.h"

#include <cstring>
#include <utility>
#include <vector>

#include "tensor/attention.h"
#include "tensor/dtype.h"
#include "tensor/tensor_ops.h"

namespace chibillm {
namespace {

result<qwen_qkv, tensor_op_error>
apply_rope(const metal_context& context,
           const qwen3_config& config,
           qwen_qkv qkv,
           const metal_tensor& positions)
{
    auto query = rope(context, qkv.query, positions, config.query_head_count, config.rope_theta);
    if (!query)
        return fail(query.error());
    auto key = rope(context, qkv.key, positions, config.kv_head_count, config.rope_theta);
    if (!key)
        return fail(key.error());
    return qwen_qkv { std::move(*query), std::move(*key), std::move(qkv.value) };
}

result<metal_tensor, tensor_op_error>
run_attention(const metal_context& context,
              const qwen3_config& config,
              const qwen_layer_weights& weights,
              std::size_t layer,
              const metal_tensor& hidden_states,
              qwen_qkv qkv,
              const metal_tensor& slots,
              const prepared_attention_batch& prepared,
              metal_kv_cache& cache)
{
    CL_TRY(store_kv(context, qkv.key, qkv.value, slots, layer, cache));

    auto attended =
        paged_attention(context, qkv.query, prepared, layer, config.query_head_count, cache);
    if (!attended)
        return fail(attended.error());
    return linear_add(context, *attended, weights.attention_output, hidden_states);
}

} // namespace

result<qwen_qkv, tensor_op_error>
project_qwen_qkv(const metal_context& context,
                 const qwen3_config& config,
                 const qwen_layer_weights& weights,
                 const metal_tensor& hidden_states)
{
    const auto& shape = hidden_states.descriptor().shape();
    if (shape.rank() != 2
        || hidden_states.descriptor().type() != dtype::f32
        || shape.dimensions()[1] != config.hidden_size) {
        return fail(tensor_op_errc::input_shape_mismatch);
    }

    auto normalized = rms_norm(context, hidden_states, weights.input_norm, config.rms_epsilon);
    if (!normalized)
        return fail(normalized.error());
    auto projected = linear_split(context, *normalized, weights.qkv_packed, config.query_width(),
                                  config.kv_width(), config.kv_width());
    if (!projected)
        return fail(projected.error());
    auto& [query, key, value] = *projected;
    return qwen_qkv { std::move(query), std::move(key), std::move(value) };
}

result<qwen_qkv, tensor_op_error>
normalize_qwen_qk(const metal_context& context,
                  const qwen3_config& config,
                  const qwen_layer_weights& weights,
                  qwen_qkv qkv)
{
    const auto& query_shape = qkv.query.descriptor().shape();
    if (query_shape.rank() != 2) {
        return fail(tensor_op_errc::input_shape_mismatch);
    }

    const auto& key_shape = qkv.key.descriptor().shape();
    if (key_shape.rank() != 2)
        return fail(tensor_op_errc::invalid_rank);
    if (query_shape.dimensions()[1] != config.query_width()
        || key_shape.dimensions()[0] != query_shape.dimensions()[0]
        || key_shape.dimensions()[1] != config.kv_width())
        return fail(tensor_op_errc::output_shape_mismatch);

    auto query = rms_norm(context, qkv.query, weights.query_norm, config.rms_epsilon);
    if (!query)
        return fail(query.error());
    auto key = rms_norm(context, qkv.key, weights.key_norm, config.rms_epsilon);
    if (!key)
        return fail(key.error());

    return qwen_qkv { std::move(*query), std::move(*key), std::move(qkv.value) };
}

result<metal_tensor, tensor_op_error>
run_qwen_layers(const metal_context& context,
                const qwen3_config& config,
                const qwen_weights& weights,
                metal_tensor hidden_states,
                attention_metadata metadata,
                metal_kv_cache& cache)
{
    if (weights.layers.size() != config.layer_count) {
        return fail(tensor_op_errc::input_shape_mismatch);
    }
    auto uploaded = upload_attention_metadata(context, metadata);
    if (!uploaded)
        return fail(uploaded.error());

    auto prepared =
        prepared_attention_batch::make(context, uploaded->positions, uploaded->block_table,
                                       uploaded->table_offsets, uploaded->table_lengths, cache);
    if (!prepared)
        return fail(prepared.error());

    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        const auto& layer_weights = weights.layers[layer];
        auto qkv = project_qwen_qkv(context, config, layer_weights, hidden_states);
        if (!qkv)
            return fail(at_stage(qkv.error(), "layer " + std::to_string(layer)));
        qkv = normalize_qwen_qk(context, config, layer_weights, std::move(*qkv));
        if (!qkv)
            return fail(at_stage(qkv.error(), "layer " + std::to_string(layer)));
        qkv = apply_rope(context, config, std::move(*qkv), prepared->positions());
        if (!qkv)
            return fail(at_stage(qkv.error(), "layer " + std::to_string(layer)));

        auto attention = run_attention(context, config, layer_weights, layer, hidden_states,
                                       std::move(*qkv), uploaded->slots, *prepared, cache);
        if (!attention)
            return fail(at_stage(attention.error(), "layer " + std::to_string(layer)));
        auto output = normalized_swiglu(context, layer_weights.post_attention_norm,
                                        layer_weights.gateup_packed, layer_weights.mlp_down,
                                        config.rms_epsilon, *attention);
        if (!output)
            return fail(at_stage(output.error(), "layer " + std::to_string(layer)));
        hidden_states = std::move(*output);
    }

    return std::move(hidden_states);
}

result<metal_tensor, tensor_op_error>
run_qwen3_5_full_attention(const metal_context& context,
                           const qwen3_5_config& config,
                           const qwen3_5_layer_weights& weights,
                           const metal_tensor& hidden_states,
                           const metal_tensor& slots,
                           const prepared_attention_batch& prepared,
                           std::size_t cache_layer,
                           metal_kv_cache& cache)
{
    const auto* attention = std::get_if<qwen3_5_full_attention_weights>(&weights.mixer);
    const auto& shape = hidden_states.descriptor().shape();
    if (!attention
        || !config.attention_output_gate
        || shape.rank() != 2
        || shape.dimensions()[1] != config.hidden_size
        || config.kv_head_count != cache.kv_head_count()
        || config.head_dimension != cache.head_dimension()
        || config.query_head_count == 0
        || config.query_head_count % cache.kv_head_count() != 0
        || cache_layer >= cache.layer_count()
        || config.rotary_dimension() == 0)
        return fail(tensor_op_errc::input_shape_mismatch);

    auto normalized =
        rms_norm(context, hidden_states, weights.input_norm, config.rms_epsilon, true);
    if (!normalized)
        return fail(at_stage(normalized.error(), "cache layer " + std::to_string(cache_layer)));
    auto projected = linear_split(context, *normalized, attention->qkv_packed,
                                  2 * config.query_width(), config.kv_width(), config.kv_width());
    if (!projected)
        return fail(at_stage(projected.error(), "cache layer " + std::to_string(cache_layer)));
    auto& [query_gate, key, value] = *projected;
    auto split = split_heads(context, query_gate, config.query_head_count);
    if (!split)
        return fail(at_stage(split.error(), "cache layer " + std::to_string(cache_layer)));
    auto& [query, gate] = *split;

    // Each norm and rotation supports in-place operation on independent heads.
    CL_TRY(rms_norm(context, query, attention->query_norm, config.rms_epsilon, query, true));
    CL_TRY(rms_norm(context, key, attention->key_norm, config.rms_epsilon, key, true));
    CL_TRY(rope(context, query, prepared.positions(), config.query_head_count, config.rope_theta,
                query, config.rotary_dimension()));
    CL_TRY(rope(context, key, prepared.positions(), config.kv_head_count, config.rope_theta, key,
                config.rotary_dimension()));
    CL_TRY(store_kv(context, key, value, slots, cache_layer, cache));

    auto attended =
        paged_attention(context, query, prepared, cache_layer, config.query_head_count, cache);
    if (!attended)
        return fail(at_stage(attended.error(), "cache layer " + std::to_string(cache_layer)));
    CL_TRY(sigmoid_mul(context, gate, *attended, *attended));

    auto output = linear_add(context, *attended, attention->output, hidden_states);
    if (!output)
        return fail(at_stage(output.error(), "cache layer " + std::to_string(cache_layer)));
    return output;
}

} // namespace chibillm
