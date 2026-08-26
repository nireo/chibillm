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

    auto shape = tensor_shape::make(info->shape);
    if (!shape) {
        return fail(qwen_weights_errc::tensor_creation_failed);
    }
    auto descriptor = tensor_descriptor::make(dtype::bf16, std::move(*shape));
    if (!descriptor) {
        return fail(qwen_weights_errc::tensor_creation_failed);
    }
    auto tensor = metal_tensor::make(context, std::move(*descriptor));
    if (!tensor) {
        return fail(qwen_weights_errc::metal_allocation_failed);
    }
    if (!file.read(name, tensor->buffer().bytes())) {
        return fail(qwen_weights_errc::tensor_read_failed);
    }
    return std::move(*tensor);
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
    auto query = load("self_attn.q_proj.weight");
    if (!query)
        return fail(query.error());
    auto key = load("self_attn.k_proj.weight");
    if (!key)
        return fail(key.error());
    auto value = load("self_attn.v_proj.weight");
    if (!value)
        return fail(value.error());
    auto attention_output = load("self_attn.o_proj.weight");
    if (!attention_output)
        return fail(attention_output.error());
    auto mlp_gate = load("mlp.gate_proj.weight");
    if (!mlp_gate)
        return fail(mlp_gate.error());
    auto mlp_up = load("mlp.up_proj.weight");
    if (!mlp_up)
        return fail(mlp_up.error());
    auto mlp_down = load("mlp.down_proj.weight");
    if (!mlp_down)
        return fail(mlp_down.error());

    return qwen_layer_weights {
        std::move(*input_norm), std::move(*post_attention_norm),
        std::move(*query_norm), std::move(*key_norm),
        std::move(*query),      std::move(*key),
        std::move(*value),      std::move(*attention_output),
        std::move(*mlp_gate),   std::move(*mlp_up),
        std::move(*mlp_down),
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
