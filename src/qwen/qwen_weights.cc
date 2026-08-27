#include "qwen/qwen_weights.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace chibillm {
namespace {

result<void, qwen_weights_errc>
expect_tensor(const safetensors_file& weights,
              std::string_view name,
              std::initializer_list<std::size_t> shape)
{
    const auto* tensor = weights.find(name);
    if (tensor == nullptr) {
        return fail(qwen_weights_errc::missing_tensor);
    }
    if (tensor->type != safetensors_dtype::bf16) {
        return fail(qwen_weights_errc::unsupported_dtype);
    }
    if (tensor->shape.size() != shape.size()
        || !std::equal(tensor->shape.begin(), tensor->shape.end(), shape.begin())) {
        return fail(qwen_weights_errc::tensor_shape_mismatch);
    }
    return {};
}

std::string
layer_tensor_name(std::size_t layer, std::string_view suffix)
{
    return "model.layers." + std::to_string(layer) + "." + std::string(suffix);
}

result<void, qwen_weights_errc>
validate_layer(const safetensors_file& weights, const qwen_config& config, std::size_t layer)
{
    const auto hidden = config.hidden_size;
    const auto intermediate = config.intermediate_size;
    const auto query_width = config.query_width();
    const auto kv_width = config.kv_width();
    const auto head_dimension = config.head_dimension;

    const auto expect = [&](std::string_view suffix, std::initializer_list<std::size_t> shape) {
        return expect_tensor(weights, layer_tensor_name(layer, suffix), shape);
    };

    auto checked = expect("input_layernorm.weight", { hidden });
    if (!checked) {
        return checked;
    }
    checked = expect("post_attention_layernorm.weight", { hidden });
    if (!checked) {
        return checked;
    }
    checked = expect("self_attn.q_norm.weight", { head_dimension });
    if (!checked) {
        return checked;
    }
    checked = expect("self_attn.k_norm.weight", { head_dimension });
    if (!checked) {
        return checked;
    }
    checked = expect("self_attn.q_proj.weight", { query_width, hidden });
    if (!checked) {
        return checked;
    }
    checked = expect("self_attn.k_proj.weight", { kv_width, hidden });
    if (!checked) {
        return checked;
    }
    checked = expect("self_attn.v_proj.weight", { kv_width, hidden });
    if (!checked) {
        return checked;
    }
    checked = expect("self_attn.o_proj.weight", { hidden, query_width });
    if (!checked) {
        return checked;
    }
    checked = expect("mlp.gate_proj.weight", { intermediate, hidden });
    if (!checked) {
        return checked;
    }
    checked = expect("mlp.up_proj.weight", { intermediate, hidden });
    if (!checked) {
        return checked;
    }
    return expect("mlp.down_proj.weight", { hidden, intermediate });
}

result<metal_tensor, qwen_weights_errc>
load_tensor(const metal_context& context, const safetensors_file& file, std::string_view name)
{
    const auto* info = file.find(name);
    if (info == nullptr) {
        return fail(qwen_weights_errc::missing_tensor);
    }

    auto tensor = metal_tensor::make(context, dtype::bf16, info->shape);
    if (!tensor) {
        return fail(tensor.error() == metal_tensor_errc::invalid_descriptor
                        ? qwen_weights_errc::tensor_creation_failed
                        : qwen_weights_errc::metal_allocation_failed);
    }
    if (!file.read(name, tensor->buffer().bytes())) {
        return fail(qwen_weights_errc::tensor_read_failed);
    }
    return std::move(*tensor);
}

// loads row-major [out, in] bf16 tensors back-to-back into one metal tensor so
// downstream code can run a single projection over them.
result<metal_tensor, qwen_weights_errc>
load_packed_tensors(const metal_context& context,
                    const safetensors_file& file,
                    const std::vector<std::string>& names)
{
    const auto* first = file.find(names[0]);
    if (first == nullptr || first->shape.size() != 2) {
        return fail(qwen_weights_errc::missing_tensor);
    }
    const auto inner_features = first->shape[1];

    std::size_t total_rows = 0;
    for (const auto& name : names) {
        const auto* info = file.find(name);
        if (info == nullptr
            || info->type != safetensors_dtype::bf16
            || info->shape.size() != 2
            || info->shape[1] != inner_features) {
            return fail(qwen_weights_errc::tensor_shape_mismatch);
        }
        if (total_rows > std::numeric_limits<std::size_t>::max() - info->shape[0]) {
            return fail(qwen_weights_errc::tensor_count_overflow);
        }
        total_rows += info->shape[0];
    }

    auto packed = metal_tensor::make(context, dtype::bf16, { total_rows, inner_features });
    if (!packed) {
        return fail(packed.error() == metal_tensor_errc::invalid_descriptor
                        ? qwen_weights_errc::tensor_creation_failed
                        : qwen_weights_errc::metal_allocation_failed);
    }

    const auto element_size = sizeof(std::uint16_t);
    auto bytes = packed->buffer().bytes();
    std::size_t row_offset = 0;
    for (const auto& name : names) {
        const auto* info = file.find(name);
        auto destination =
            bytes.subspan(row_offset * inner_features * element_size,
                          static_cast<std::size_t>(info->shape[0]) * inner_features * element_size);
        if (!file.read(name, destination)) {
            return fail(qwen_weights_errc::tensor_read_failed);
        }
        row_offset += info->shape[0];
    }

    return std::move(*packed);
}

result<qwen_layer_weights, qwen_weights_errc>
load_layer(const metal_context& context, const safetensors_file& file, std::size_t layer)
{
    const auto load = [&](std::string_view suffix) {
        return load_tensor(context, file, layer_tensor_name(layer, suffix));
    };

    auto input_norm = load("input_layernorm.weight");
    if (!input_norm)
        return fail(input_norm.error());
    auto post_attention_norm = load("post_attention_layernorm.weight");
    if (!post_attention_norm)
        return fail(post_attention_norm.error());
    auto query_norm = load("self_attn.q_norm.weight");
    if (!query_norm)
        return fail(query_norm.error());
    auto key_norm = load("self_attn.k_norm.weight");
    if (!key_norm)
        return fail(key_norm.error());

    auto qkv_packed = load_packed_tensors(context, file,
                                          { layer_tensor_name(layer, "self_attn.q_proj.weight"),
                                            layer_tensor_name(layer, "self_attn.k_proj.weight"),
                                            layer_tensor_name(layer, "self_attn.v_proj.weight") });
    if (!qkv_packed)
        return fail(qkv_packed.error());
    auto attention_output = load("self_attn.o_proj.weight");
    if (!attention_output)
        return fail(attention_output.error());
    auto gateup_packed = load_packed_tensors(context, file,
                                             { layer_tensor_name(layer, "mlp.gate_proj.weight"),
                                               layer_tensor_name(layer, "mlp.up_proj.weight") });
    if (!gateup_packed)
        return fail(gateup_packed.error());
    auto mlp_down = load("mlp.down_proj.weight");
    if (!mlp_down)
        return fail(mlp_down.error());

    return qwen_layer_weights {
        std::move(*input_norm),    std::move(*post_attention_norm),
        std::move(*query_norm),    std::move(*key_norm),
        std::move(*qkv_packed),    std::move(*attention_output),
        std::move(*gateup_packed), std::move(*mlp_down),
    };
}

} // namespace

result<void, qwen_weights_errc>
validate_qwen_weights(const safetensors_file& weights, const qwen_config& config)
{
    constexpr std::size_t tensors_per_layer = 11;
    constexpr std::size_t global_tensors = 3;
    if (config.layer_count
        > (std::numeric_limits<std::size_t>::max() - global_tensors) / tensors_per_layer) {
        return fail(qwen_weights_errc::tensor_count_overflow);
    }

    auto checked = expect_tensor(weights, "model.embed_tokens.weight",
                                 { config.vocabulary_size, config.hidden_size });
    if (!checked) {
        return checked;
    }
    checked = expect_tensor(weights, "model.norm.weight", { config.hidden_size });
    if (!checked) {
        return checked;
    }
    checked =
        expect_tensor(weights, "lm_head.weight", { config.vocabulary_size, config.hidden_size });
    if (!checked) {
        return checked;
    }

    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        checked = validate_layer(weights, config, layer);
        if (!checked) {
            return checked;
        }
    }

    const auto expected_count = global_tensors + config.layer_count * tensors_per_layer;
    if (weights.tensor_count() != expected_count) {
        return fail(qwen_weights_errc::unexpected_tensor_count);
    }

    return {};
}

result<qwen_weights, qwen_weights_errc>
load_qwen_weights(const metal_context& context,
                  const safetensors_file& file,
                  const qwen_config& config)
{
    auto validated = validate_qwen_weights(file, config);
    if (!validated) {
        return fail(validated.error());
    }

    auto token_embedding = load_tensor(context, file, "model.embed_tokens.weight");
    if (!token_embedding)
        return fail(token_embedding.error());
    auto final_norm = load_tensor(context, file, "model.norm.weight");
    if (!final_norm)
        return fail(final_norm.error());
    auto output = load_tensor(context, file, "lm_head.weight");
    if (!output)
        return fail(output.error());

    std::vector<qwen_layer_weights> layers;
    layers.reserve(config.layer_count);
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        auto loaded = load_layer(context, file, layer);
        if (!loaded) {
            return fail(loaded.error());
        }
        layers.push_back(std::move(*loaded));
    }

    return qwen_weights {
        std::move(*token_embedding),
        std::move(*final_norm),
        std::move(*output),
        std::move(layers),
    };
}

} // namespace chibillm
