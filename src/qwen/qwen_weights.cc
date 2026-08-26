#include "qwen/qwen_weights.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>

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

} // namespace chibillm
