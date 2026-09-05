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

result<qwen_qkv, tensor_op_errc>
apply_rope(const metal_context& context,
           const qwen3_config& config,
           qwen_qkv qkv,
           const metal_tensor& positions)
{
    const auto rows = positions.descriptor().shape().dimensions()[0];
    auto query = allocate_tensor(context, dtype::f32, { rows, config.query_width() });
    if (!query)
        return fail(query.error());
    CL_TRY(rope(context, qkv.query, positions, config.query_head_count, config.rope_theta, *query));
    auto key = allocate_tensor(context, dtype::f32, { rows, config.kv_width() });
    if (!key)
        return fail(key.error());
    CL_TRY(rope(context, qkv.key, positions, config.kv_head_count, config.rope_theta, *key));
    return qwen_qkv { std::move(*query), std::move(*key), std::move(qkv.value) };
}

result<metal_tensor, tensor_op_errc>
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

    const auto rows = prepared.positions().descriptor().shape().dimensions()[0];
    auto attended = allocate_tensor(context, dtype::f32, { rows, config.query_width() });
    if (!attended)
        return fail(attended.error());
    CL_TRY(paged_attention(context, qkv.query, prepared, layer, config.query_head_count, cache,
                           *attended));
    auto residual = allocate_tensor(context, dtype::f32, { rows, config.hidden_size });
    if (!residual)
        return fail(residual.error());
    CL_TRY(linear_add(context, *attended, weights.attention_output, hidden_states, *residual));
    return std::move(*residual);
}

} // namespace

result<qwen_qkv, tensor_op_errc>
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

    const auto rows = shape.dimensions()[0];
    auto normalized = allocate_tensor(context, dtype::f32, { rows, config.hidden_size });
    if (!normalized)
        return fail(normalized.error());
    CL_TRY(rms_norm(context, hidden_states, weights.input_norm, config.rms_epsilon, *normalized));
    auto query = allocate_tensor(context, dtype::f32, { rows, config.query_width() });
    if (!query)
        return fail(query.error());
    auto key = allocate_tensor(context, dtype::f32, { rows, config.kv_width() });
    if (!key)
        return fail(key.error());
    auto value = allocate_tensor(context, dtype::f32, { rows, config.kv_width() });
    if (!value)
        return fail(value.error());
    CL_TRY(linear_split(context, *normalized, weights.qkv_packed, { &*query, &*key, &*value }));

    return qwen_qkv { std::move(*query), std::move(*key), std::move(*value) };
}

result<qwen_qkv, tensor_op_errc>
normalize_qwen_qk(const metal_context& context,
                  const qwen3_config& config,
                  const qwen_layer_weights& weights,
                  qwen_qkv qkv)
{
    const auto& query_shape = qkv.query.descriptor().shape();
    if (query_shape.rank() != 2) {
        return fail(tensor_op_errc::input_shape_mismatch);
    }

    const auto rows = query_shape.dimensions()[0];
    auto query = allocate_tensor(context, dtype::f32, { rows, config.query_width() });
    if (!query)
        return fail(query.error());
    CL_TRY(rms_norm(context, qkv.query, weights.query_norm, config.rms_epsilon, *query));
    auto key = allocate_tensor(context, dtype::f32, { rows, config.kv_width() });
    if (!key)
        return fail(key.error());
    CL_TRY(rms_norm(context, qkv.key, weights.key_norm, config.rms_epsilon, *key));

    return qwen_qkv { std::move(*query), std::move(*key), std::move(qkv.value) };
}

result<metal_tensor, tensor_op_errc>
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
            return fail(qkv.error());
        qkv = normalize_qwen_qk(context, config, layer_weights, std::move(*qkv));
        if (!qkv)
            return fail(qkv.error());
        qkv = apply_rope(context, config, std::move(*qkv), prepared->positions());
        if (!qkv)
            return fail(qkv.error());

        auto attention = run_attention(context, config, layer_weights, layer, hidden_states,
                                       std::move(*qkv), uploaded->slots, *prepared, cache);
        if (!attention)
            return fail(attention.error());
        auto output = normalized_swiglu(context, layer_weights.post_attention_norm,
                                        layer_weights.gateup_packed, layer_weights.mlp_down,
                                        config.rms_epsilon, *attention);
        if (!output)
            return fail(output.error());
        hidden_states = std::move(*output);
    }

    return std::move(hidden_states);
}

} // namespace chibillm
