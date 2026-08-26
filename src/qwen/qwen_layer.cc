#include "qwen/qwen_layer.h"

#include <utility>

#include "tensor/dtype.h"
#include "tensor/tensor_descriptor.h"
#include "tensor/tensor_ops.h"
#include "tensor/tensor_shape.h"

namespace chibillm {
namespace {

result<metal_tensor, qwen_layer_errc>
make_tensor(const metal_context& context, std::size_t rows, std::size_t columns)
{
    auto shape = tensor_shape::make({ rows, columns });
    if (!shape) {
        return fail(qwen_layer_errc::tensor_creation_failed);
    }
    auto descriptor = tensor_descriptor::make(dtype::f32, std::move(*shape));
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
    auto normalized = make_tensor(context, rows, config.hidden_size);
    auto query = make_tensor(context, rows, config.query_width());
    auto key = make_tensor(context, rows, config.kv_width());
    auto value = make_tensor(context, rows, config.kv_width());
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

} // namespace chibillm
