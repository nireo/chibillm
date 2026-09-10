#include "qwen/qwen_layer.h"

#include "model_batch.h"
#include "qwen/qwen3_5_model_state.h"
#include "tensor/deltanet.h"
#include "tensor/layers.h"

#include <algorithm>
#include <array>
#include <unordered_set>
#include <utility>

namespace chibillm {
namespace {

result<metal_tensor, tensor_op_errc>
run_linear_attention(const metal_context& context,
                     const qwen3_5_config& config,
                     const qwen3_5_layer_weights& weights,
                     const metal_tensor& hidden_states,
                     std::span<const model_batch_item> items,
                     std::size_t layer,
                     qwen3_5_model_state& state)
{
    const auto& attention = std::get<qwen3_5_linear_attention_weights>(weights.mixer);
    const auto rows = hidden_states.descriptor().shape().dimensions()[0];
    const auto qkv_width = 2 * config.linear_key_width() + config.linear_value_width();
    const auto value_width = config.linear_value_width();
    auto normalized = allocate_tensor(context, dtype::f32, { rows, config.hidden_size });
    if (!normalized)
        return fail(normalized.error());
    CL_TRY(rms_norm(context, hidden_states, weights.input_norm, config.rms_epsilon, *normalized,
                    true));

    auto qkv = allocate_tensor(context, dtype::f32, { rows, qkv_width });
    if (!qkv)
        return fail(qkv.error());
    auto gate = allocate_tensor(context, dtype::f32, { rows, value_width });
    if (!gate)
        return fail(gate.error());
    auto a = allocate_tensor(context, dtype::f32, { rows, config.linear_value_head_count });
    if (!a)
        return fail(a.error());
    auto b = allocate_tensor(context, dtype::f32, { rows, config.linear_value_head_count });
    if (!b)
        return fail(b.error());
    CL_TRY(linear_split(context, *normalized, attention.qkv_projection, { &*qkv }));
    CL_TRY(linear_split(context, *normalized, attention.gate_projection, { &*gate }));
    CL_TRY(linear_split(context, *normalized, attention.decay_projection, { &*a }));
    CL_TRY(linear_split(context, *normalized, attention.learning_rate_projection, { &*b }));

    auto convolved = allocate_tensor(context, dtype::f32, { rows, qkv_width });
    if (!convolved)
        return fail(convolved.error());
    auto mixed = allocate_tensor(context, dtype::f32, { rows, value_width });
    if (!mixed)
        return fail(mixed.error());
    // Only the causal scans are sequence-local. Buffer offsets select each
    // chunk without copying or synchronizing the batched GPU activations.
    for (const auto& item : items) {
        auto& memory = *state.linear_state(item.id, layer); // Validated before encoding any layer.
        const deltanet_chunk chunk { item.token_offset, item.token_count };
        CL_TRY(causal_conv1d_silu(context, *qkv, attention.convolution, memory.convolution,
                                  *convolved, chunk));
        CL_TRY(gated_delta_rule(context, *convolved, *a, *b, attention.decay_log,
                                attention.learning_rate_bias, config.linear_key_head_count,
                                memory.recurrent, *mixed, 1e-6F, chunk));
    }

    auto gated = allocate_tensor(context, dtype::f32, { rows, value_width });
    if (!gated)
        return fail(gated.error());
    CL_TRY(rms_norm_gated(context, *mixed, *gate, attention.norm, config.rms_epsilon, *gated));
    auto output = allocate_tensor(context, dtype::f32, { rows, config.hidden_size });
    if (!output)
        return fail(output.error());
    CL_TRY(linear_add(context, *gated, attention.output, hidden_states, *output));
    return std::move(*output);
}

// Validate routing before any persistent state can be mutated. Shape checks for
// individual weights remain in the tensor ops; a later failure requires abort.
result<void, tensor_op_errc>
validate_routing(const qwen3_5_config& config,
                 const qwen3_5_weights& weights,
                 const model_batch& batch,
                 const qwen3_5_model_state& state)
{
    if (batch.phase != batch_phase::prefill && batch.phase != batch_phase::decode)
        return fail(tensor_op_errc::input_shape_mismatch);
    std::unordered_set<seq_id> seen;
    for (const auto& item : batch.items) {
        const auto committed = state.committed_tokens(item.id);
        if (!committed
            || !seen.insert(item.id).second
            || !std::ranges::equal(item.block_table, state.resources(item.id).blocks)
            || (batch.phase == batch_phase::decode && item.token_count != 1))
            return fail(tensor_op_errc::input_shape_mismatch);
        for (std::size_t i = 0; i < item.token_count; ++i) {
            if (batch.positions[item.token_offset + i] != *committed + i)
                return fail(tensor_op_errc::input_shape_mismatch);
        }
    }

    std::size_t full_count = 0, linear_count = 0;
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        const auto& mixer = weights.layers[layer].mixer;
        switch (config.layer_types[layer]) {
        case qwen3_5_layer_type::linear_attention:
            if (!std::holds_alternative<qwen3_5_linear_attention_weights>(mixer)
                || state.cache_layer(layer))
                return fail(tensor_op_errc::input_shape_mismatch);
            for (const auto& item : batch.items) {
                const auto* memory = state.linear_state(item.id, layer);
                if (!memory
                    || !std::ranges::equal(
                        memory->convolution.descriptor().shape().dimensions(),
                        std::array { 2 * config.linear_key_width() + config.linear_value_width(),
                                     config.linear_conv_kernel_dimension })
                    || !std::ranges::equal(memory->recurrent.descriptor().shape().dimensions(),
                                           std::array { config.linear_value_head_count,
                                                        config.linear_key_head_dimension,
                                                        config.linear_value_head_dimension }))
                    return fail(tensor_op_errc::input_shape_mismatch);
            }
            ++linear_count;
            break;
        case qwen3_5_layer_type::full_attention:
            if (!std::holds_alternative<qwen3_5_full_attention_weights>(mixer)
                || state.cache_layer(layer) != full_count)
                return fail(tensor_op_errc::input_shape_mismatch);
            ++full_count;
            break;
        default:
            return fail(tensor_op_errc::input_shape_mismatch);
        }
    }
    if (full_count != state.cache().layer_count()
        || linear_count != state.linear_layer_count()
        || config.kv_head_count != state.cache().kv_head_count()
        || config.head_dimension != state.cache().head_dimension())
        return fail(tensor_op_errc::input_shape_mismatch);
    return {};
}

} // namespace

result<metal_tensor, tensor_op_errc>
run_qwen3_5_layers(const metal_context& context,
                   const qwen3_5_config& config,
                   const qwen3_5_weights& weights,
                   metal_tensor hidden_states,
                   const model_batch& batch,
                   qwen3_5_model_state& state)
{
    const auto& shape = hidden_states.descriptor().shape();
    if (!config.layer_count
        || weights.layers.size() != config.layer_count
        || config.layer_types.size() != config.layer_count
        || hidden_states.descriptor().type() != dtype::f32
        || shape.rank() != 2
        || shape.dimensions()[0] != batch.token_count()
        || shape.dimensions()[1] != config.hidden_size)
        return fail(tensor_op_errc::input_shape_mismatch);

    auto metadata = prepare_paged_batch(batch, config.max_position_embeddings,
                                        state.cache().block_count(), state.block_size());
    if (!metadata)
        return fail(tensor_op_errc::input_shape_mismatch);
    CL_TRY(validate_routing(config, weights, batch, state));
    auto uploaded =
        upload_attention_metadata(context,
                                  { batch.positions, metadata->slots, metadata->block_table,
                                    metadata->table_offsets, metadata->table_lengths });
    if (!uploaded)
        return fail(uploaded.error());
    auto prepared = prepared_attention_batch::make(context, uploaded->positions,
                                                   uploaded->block_table, uploaded->table_offsets,
                                                   uploaded->table_lengths, state.cache());
    if (!prepared)
        return fail(prepared.error());

    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        const auto& layer_weights = weights.layers[layer];
        auto mixed = config.layer_types[layer] == qwen3_5_layer_type::linear_attention
            ? run_linear_attention(context, config, layer_weights, hidden_states, batch.items,
                                   layer, state)
            : run_qwen3_5_full_attention(context, config, layer_weights, hidden_states,
                                         uploaded->slots, *prepared, *state.cache_layer(layer),
                                         state.cache());
        if (!mixed)
            return fail(mixed.error());
        auto output = normalized_swiglu(context, layer_weights.post_attention_norm,
                                        layer_weights.gateup_packed, layer_weights.mlp_down,
                                        config.rms_epsilon, *mixed, true);
        if (!output)
            return fail(output.error());
        hidden_states = std::move(*output);
    }
    return std::move(hidden_states);
}

} // namespace chibillm
