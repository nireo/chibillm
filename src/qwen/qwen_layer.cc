#include "qwen/qwen_layer.h"

#include <cstring>
#include <utility>
#include <vector>

#include "tensor/dtype.h"
#include "tensor/tensor_descriptor.h"
#include "tensor/tensor_ops.h"
#include "tensor/tensor_shape.h"

namespace chibillm {
namespace {

result<metal_tensor, qwen_layer_errc>
make_tensor(const metal_context& context, dtype type, std::vector<std::size_t> dimensions)
{
    auto shape = tensor_shape::make(std::move(dimensions));
    if (!shape) {
        return fail(qwen_layer_errc::tensor_creation_failed);
    }
    auto descriptor = tensor_descriptor::make(type, std::move(*shape));
    if (!descriptor) {
        return fail(qwen_layer_errc::tensor_creation_failed);
    }
    auto tensor = metal_tensor::make(context, std::move(*descriptor));
    if (!tensor) {
        return fail(qwen_layer_errc::metal_allocation_failed);
    }
    return std::move(*tensor);
}

qwen_layer_errc
operation_error(tensor_op_errc error)
{
    return error == tensor_op_errc::backend_failure ? qwen_layer_errc::backend_failure
                                                    : qwen_layer_errc::invalid_input;
}

} // namespace

result<qwen_qkv, qwen_layer_errc>
project_qwen_qkv(const metal_context& context,
                 const qwen_config& config,
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
    auto query = make_tensor(context, dtype::f32, { rows, config.query_width() });
    auto key = make_tensor(context, dtype::f32, { rows, config.kv_width() });
    auto value = make_tensor(context, dtype::f32, { rows, config.kv_width() });
    if (!normalized)
        return fail(normalized.error());
    if (!query)
        return fail(query.error());
    if (!key)
        return fail(key.error());
    if (!value)
        return fail(value.error());

    auto operation =
        rms_norm(context, hidden_states, weights.input_norm, config.rms_epsilon, *normalized);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    operation = linear(context, *normalized, weights.query, *query);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    operation = linear(context, *normalized, weights.key, *key);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    operation = linear(context, *normalized, weights.value, *value);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }

    return qwen_qkv { std::move(*query), std::move(*key), std::move(*value) };
}

result<qwen_qkv, qwen_layer_errc>
normalize_qwen_qk(const metal_context& context,
                  const qwen_config& config,
                  const qwen_layer_weights& weights,
                  qwen_qkv qkv)
{
    const auto& query_shape = qkv.query.descriptor().shape();
    if (query_shape.rank() != 2) {
        return fail(qwen_layer_errc::invalid_input);
    }

    const auto rows = query_shape.dimensions()[0];
    auto query = make_tensor(context, dtype::f32, { rows, config.query_width() });
    auto key = make_tensor(context, dtype::f32, { rows, config.kv_width() });
    if (!query)
        return fail(query.error());
    if (!key)
        return fail(key.error());

    auto operation =
        rms_norm_heads(context, qkv.query, weights.query_norm, config.rms_epsilon, *query);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    operation = rms_norm_heads(context, qkv.key, weights.key_norm, config.rms_epsilon, *key);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }

    return qwen_qkv { std::move(*query), std::move(*key), std::move(qkv.value) };
}

result<qwen_qkv, qwen_layer_errc>
apply_qwen_rope(const metal_context& context,
                const qwen_config& config,
                qwen_qkv qkv,
                std::span<const std::uint32_t> positions)
{
    if (positions.empty()) {
        return fail(qwen_layer_errc::invalid_input);
    }

    const auto rows = positions.size();
    auto position_tensor = make_tensor(context, dtype::u32, { rows });
    auto query = make_tensor(context, dtype::f32, { rows, config.query_width() });
    auto key = make_tensor(context, dtype::f32, { rows, config.kv_width() });
    if (!position_tensor)
        return fail(position_tensor.error());
    if (!query)
        return fail(query.error());
    if (!key)
        return fail(key.error());

    std::memcpy(position_tensor->buffer().bytes().data(), positions.data(), positions.size_bytes());
    auto operation = rope(context, qkv.query, *position_tensor, config.query_head_count,
                          config.rope_theta, *query);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }
    operation =
        rope(context, qkv.key, *position_tensor, config.kv_head_count, config.rope_theta, *key);
    if (!operation) {
        return fail(operation_error(operation.error()));
    }

    return qwen_qkv { std::move(*query), std::move(*key), std::move(qkv.value) };
}

} // namespace chibillm
