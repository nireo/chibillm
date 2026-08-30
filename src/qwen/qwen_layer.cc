#include "qwen/qwen_layer.h"

#include <cstring>
#include <utility>
#include <vector>

#include "tensor/dtype.h"
#include "tensor/tensor_ops.h"

namespace chibillm {
namespace {

qwen_layer_errc
tensor_error(metal_tensor_errc error)
{
    return error == metal_tensor_errc::invalid_descriptor
        ? qwen_layer_errc::tensor_creation_failed
        : qwen_layer_errc::metal_allocation_failed;
}

result<metal_tensor, qwen_layer_errc>
make_tensor(const metal_context& context, dtype type, std::vector<std::size_t> dimensions)
{
    auto tensor = metal_tensor::make(context, type, std::move(dimensions));
    if (!tensor) {
        return fail(tensor_error(tensor.error()));
    }
    return std::move(*tensor);
}

qwen_layer_errc
operation_error(tensor_op_errc error)
{
    return error == tensor_op_errc::backend_failure ? qwen_layer_errc::backend_failure
                                                    : qwen_layer_errc::invalid_input;
}

result<metal_tensor, qwen_layer_errc>
upload_u32(const metal_context& context, std::span<const std::uint32_t> values)
{
    if (values.empty()) {
        return fail(qwen_layer_errc::invalid_input);
    }
    auto tensor = make_tensor(context, dtype::u32, { values.size() });
    if (!tensor) {
        return fail(tensor.error());
    }
    std::memcpy(tensor->buffer().bytes().data(), values.data(), values.size_bytes());
    return std::move(*tensor);
}

struct attention_metadata_tensors {
    metal_tensor positions;
    metal_tensor slots;
    metal_tensor block_table;
    metal_tensor table_offsets;
    metal_tensor table_lengths;
};

result<attention_metadata_tensors, qwen_layer_errc>
upload_attention_metadata(const metal_context& context, qwen_attention_metadata metadata)
{
    auto positions = upload_u32(context, metadata.positions);
    if (!positions)
        return fail(positions.error());
    auto slots = upload_u32(context, metadata.slots);
    if (!slots)
        return fail(slots.error());
    auto block_table = upload_u32(context, metadata.block_table);
    if (!block_table)
        return fail(block_table.error());
    auto table_offsets = upload_u32(context, metadata.block_table_offsets);
    if (!table_offsets)
        return fail(table_offsets.error());
    auto table_lengths = upload_u32(context, metadata.block_table_lengths);
    if (!table_lengths)
        return fail(table_lengths.error());
    return attention_metadata_tensors { std::move(*positions), std::move(*slots),
                                        std::move(*block_table), std::move(*table_offsets),
                                        std::move(*table_lengths) };
}

result<qwen_qkv, qwen_layer_errc>
apply_rope(const metal_context& context,
           const qwen3_config& config,
           qwen_qkv qkv,
           const metal_tensor& positions)
{
    const auto rows = positions.descriptor().shape().dimensions()[0];
    auto query = make_tensor(context, dtype::f32, { rows, config.query_width() });
    if (!query)
        return fail(query.error());
    auto operation =
        rope(context, qkv.query, positions, config.query_head_count, config.rope_theta, *query);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    auto key = make_tensor(context, dtype::f32, { rows, config.kv_width() });
    if (!key)
        return fail(key.error());
    operation = rope(context, qkv.key, positions, config.kv_head_count, config.rope_theta, *key);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    return qwen_qkv { std::move(*query), std::move(*key), std::move(qkv.value) };
}

result<metal_tensor, qwen_layer_errc>
run_attention(const metal_context& context,
              const qwen3_config& config,
              const qwen_layer_weights& weights,
              std::size_t layer,
              const metal_tensor& hidden_states,
              qwen_qkv qkv,
              const attention_metadata_tensors& metadata,
              metal_kv_cache& cache)
{
    auto operation = store_kv(context, qkv.key, qkv.value, metadata.slots, layer, cache);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }

    const auto rows = metadata.positions.descriptor().shape().dimensions()[0];
    auto attended = make_tensor(context, dtype::f32, { rows, config.query_width() });
    if (!attended)
        return fail(attended.error());
    operation = paged_attention(context, qkv.query, metadata.positions, metadata.block_table,
                                metadata.table_offsets, metadata.table_lengths, layer,
                                config.query_head_count, cache, *attended);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    auto residual = make_tensor(context, dtype::f32, { rows, config.hidden_size });
    if (!residual)
        return fail(residual.error());
    operation = linear_add(context, *attended, weights.attention_output, hidden_states, *residual);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    return std::move(*residual);
}

} // namespace

result<qwen_qkv, qwen_layer_errc>
project_qwen_qkv(const metal_context& context,
                 const qwen3_config& config,
                 const qwen_layer_weights& weights,
                 const metal_tensor& hidden_states)
{
    const auto& shape = hidden_states.descriptor().shape();
    if (shape.rank() != 2
        || hidden_states.descriptor().type() != dtype::f32
        || shape.dimensions()[1] != config.hidden_size) {
        return fail(qwen_layer_errc::invalid_input);
    }

    const auto rows = shape.dimensions()[0];
    auto normalized = make_tensor(context, dtype::f32, { rows, config.hidden_size });
    if (!normalized)
        return fail(normalized.error());
    auto operation =
        rms_norm(context, hidden_states, weights.input_norm, config.rms_epsilon, *normalized);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    auto query = make_tensor(context, dtype::f32, { rows, config.query_width() });
    if (!query)
        return fail(query.error());
    auto key = make_tensor(context, dtype::f32, { rows, config.kv_width() });
    if (!key)
        return fail(key.error());
    auto value = make_tensor(context, dtype::f32, { rows, config.kv_width() });
    if (!value)
        return fail(value.error());
    operation = linear_split(context, *normalized, weights.qkv_packed, { &*query, &*key, &*value });
    if (!operation) {
        return fail(operation_error(operation.error()));
    }

    return qwen_qkv { std::move(*query), std::move(*key), std::move(*value) };
}

result<qwen_qkv, qwen_layer_errc>
normalize_qwen_qk(const metal_context& context,
                  const qwen3_config& config,
                  const qwen_layer_weights& weights,
                  qwen_qkv qkv)
{
    const auto& query_shape = qkv.query.descriptor().shape();
    if (query_shape.rank() != 2) {
        return fail(qwen_layer_errc::invalid_input);
    }

    const auto rows = query_shape.dimensions()[0];
    auto query = make_tensor(context, dtype::f32, { rows, config.query_width() });
    if (!query)
        return fail(query.error());
    auto operation =
        rms_norm_heads(context, qkv.query, weights.query_norm, config.rms_epsilon, *query);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    auto key = make_tensor(context, dtype::f32, { rows, config.kv_width() });
    if (!key)
        return fail(key.error());
    operation = rms_norm_heads(context, qkv.key, weights.key_norm, config.rms_epsilon, *key);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }

    return qwen_qkv { std::move(*query), std::move(*key), std::move(qkv.value) };
}

result<qwen_qkv, qwen_layer_errc>
apply_qwen_rope(const metal_context& context,
                const qwen3_config& config,
                qwen_qkv qkv,
                std::span<const std::uint32_t> positions)
{
    if (positions.empty()) {
        return fail(qwen_layer_errc::invalid_input);
    }

    auto position_tensor = upload_u32(context, positions);
    if (!position_tensor)
        return fail(position_tensor.error());
    return apply_rope(context, config, std::move(qkv), *position_tensor);
}

result<metal_tensor, qwen_layer_errc>
run_qwen_attention(const metal_context& context,
                   const qwen3_config& config,
                   const qwen_layer_weights& weights,
                   std::size_t layer,
                   const metal_tensor& hidden_states,
                   qwen_qkv qkv,
                   qwen_attention_metadata metadata,
                   metal_kv_cache& cache)
{
    auto uploaded = upload_attention_metadata(context, metadata);
    if (!uploaded)
        return fail(uploaded.error());
    return run_attention(context, config, weights, layer, hidden_states, std::move(qkv), *uploaded,
                         cache);
}

result<metal_tensor, qwen_layer_errc>
run_qwen_mlp(const metal_context& context,
             const qwen3_config& config,
             const qwen_layer_weights& weights,
             const metal_tensor& hidden_states)
{
    const auto& shape = hidden_states.descriptor().shape();
    if (shape.rank() != 2
        || hidden_states.descriptor().type() != dtype::f32
        || shape.dimensions()[1] != config.hidden_size) {
        return fail(qwen_layer_errc::invalid_input);
    }

    const auto rows = shape.dimensions()[0];
    auto normalized = make_tensor(context, dtype::f32, { rows, config.hidden_size });
    if (!normalized)
        return fail(normalized.error());
    auto operation = rms_norm(context, hidden_states, weights.post_attention_norm,
                              config.rms_epsilon, *normalized);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    auto gate = make_tensor(context, dtype::f32, { rows, config.intermediate_size });
    if (!gate)
        return fail(gate.error());
    auto up = make_tensor(context, dtype::f32, { rows, config.intermediate_size });
    if (!up)
        return fail(up.error());
    operation = linear_split(context, *normalized, weights.gateup_packed, { &*gate, &*up });
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    auto activated = make_tensor(context, dtype::f32, { rows, config.intermediate_size });
    if (!activated)
        return fail(activated.error());
    operation = silu_mul(context, *gate, *up, *activated);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    auto residual = make_tensor(context, dtype::f32, { rows, config.hidden_size });
    if (!residual)
        return fail(residual.error());
    operation = linear_add(context, *activated, weights.mlp_down, hidden_states, *residual);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }

    return std::move(*residual);
}

result<metal_tensor, qwen_layer_errc>
run_qwen_layers(const metal_context& context,
                const qwen3_config& config,
                const qwen_weights& weights,
                metal_tensor hidden_states,
                qwen_attention_metadata metadata,
                metal_kv_cache& cache)
{
    if (weights.layers.size() != config.layer_count) {
        return fail(qwen_layer_errc::invalid_input);
    }
    auto uploaded = upload_attention_metadata(context, metadata);
    if (!uploaded)
        return fail(uploaded.error());

    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        const auto& layer_weights = weights.layers[layer];
        auto qkv = project_qwen_qkv(context, config, layer_weights, hidden_states);
        if (!qkv)
            return fail(qkv.error());
        qkv = normalize_qwen_qk(context, config, layer_weights, std::move(*qkv));
        if (!qkv)
            return fail(qkv.error());
        qkv = apply_rope(context, config, std::move(*qkv), uploaded->positions);
        if (!qkv)
            return fail(qkv.error());

        auto attention = run_attention(context, config, layer_weights, layer, hidden_states,
                                       std::move(*qkv), *uploaded, cache);
        if (!attention)
            return fail(attention.error());
        auto output = run_qwen_mlp(context, config, layer_weights, *attention);
        if (!output)
            return fail(output.error());
        hidden_states = std::move(*output);
    }

    return std::move(hidden_states);
}

} // namespace chibillm
