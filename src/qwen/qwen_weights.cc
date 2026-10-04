#include "qwen/qwen_weights.h"
#include <cstring>
#include <limits>

namespace chibillm {
namespace {
weight_layout
common_layer(std::size_t hidden, std::size_t intermediate)
{
    return {
        { "input_norm", { { "input_layernorm.weight", { hidden } } } },
        { "post_norm", { { "post_attention_layernorm.weight", { hidden } } } },
        { "gateup",
          { { "mlp.gate_proj.weight", { intermediate, hidden } },
            { "mlp.up_proj.weight", { intermediate, hidden } } },
          weight_role::matrix },
        { "down", { { "mlp.down_proj.weight", { hidden, intermediate } } }, weight_role::matrix },
    };
}

weight_layout
full_attention(std::size_t hidden, std::size_t query, std::size_t kv, std::size_t head, bool gated)
{
    return {
        { "query_norm", { { "self_attn.q_norm.weight", { head } } } },
        { "key_norm", { { "self_attn.k_norm.weight", { head } } } },
        { "qkv",
          { { "self_attn.q_proj.weight", { (gated ? 2 : 1) * query, hidden } },
            { "self_attn.k_proj.weight", { kv, hidden } },
            { "self_attn.v_proj.weight", { kv, hidden } } },
          weight_role::matrix },
        { "output", { { "self_attn.o_proj.weight", { hidden, query } } }, weight_role::matrix },
    };
}

weight_layout
linear_attention(const qwen3_5_config& config, bool converted = false)
{
    const auto hidden = config.hidden_size;
    const auto qkv = 2 * config.linear_key_width() + config.linear_value_width();
    const auto heads = config.linear_value_head_count;
    return {
        { "qkv", { { "linear_attn.in_proj_qkv.weight", { qkv, hidden } } }, weight_role::matrix },
        { "gate",
          { { "linear_attn.in_proj_z.weight", { config.linear_value_width(), hidden } } },
          weight_role::matrix },
        { "decay",
          { { "linear_attn.in_proj_a.weight", { heads, hidden } } },
          weight_role::sensitive_matrix },
        { "rate",
          { { "linear_attn.in_proj_b.weight", { heads, hidden } } },
          weight_role::sensitive_matrix },
        { "convolution",
          { { "linear_attn.conv1d.weight",
              converted
                  ? std::vector<std::size_t> { qkv, config.linear_conv_kernel_dimension, 1 }
                  : std::vector<std::size_t> { qkv, 1, config.linear_conv_kernel_dimension } } } },
        { "decay_log", { { "linear_attn.A_log", { heads }, safetensors_dtype::f32 } } },
        { "rate_bias", { { "linear_attn.dt_bias", { heads } } } },
        { "norm",
          { { "linear_attn.norm.weight",
              { config.linear_value_head_dimension },
              converted ? safetensors_dtype::bf16 : safetensors_dtype::f32 } } },
        { "output",
          { { "linear_attn.out_proj.weight", { hidden, config.linear_value_width() } } },
          weight_role::matrix },
    };
}

weight_layout
globals(std::size_t vocabulary, std::size_t hidden)
{
    weight_layout layout {
        { "embedding", { { "embed_tokens.weight", { vocabulary, hidden } } }, weight_role::matrix },
        { "norm", { { "norm.weight", { hidden } } } },
    };
    return layout;
}

weight_layout
output_layout(std::size_t vocabulary, std::size_t hidden)
{
    return { { "output", { { "lm_head.weight", { vocabulary, hidden } } }, weight_role::matrix } };
}

std::string
layer_prefix(std::string_view model, std::size_t layer)
{
    return std::string(model) + "layers." + std::to_string(layer) + ".";
}

weight_layout
mixer_layout(const qwen3_5_config& config, std::size_t layer, bool converted = false)
{
    return config.layer_types[layer] == qwen3_5_layer_type::full_attention
        ? full_attention(config.hidden_size, config.query_width(), config.kv_width(),
                         config.head_dimension, true)
        : linear_attention(config, converted);
}

struct checkpoint_layout {
    std::string_view prefix;
    bool converted;
};

checkpoint_layout
qwen3_5_layout(const safetensors_file& file, const qwen3_5_config& config)
{
    std::string_view prefix = "model.language_model.";
    for (const auto candidate : { "model.language_model.", "language_model.model.", "model." }) {
        if (file.find(std::string(candidate) + "embed_tokens.weight")) {
            prefix = candidate;
            break;
        }
    }
    bool converted = prefix != "model.language_model.";
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        if (config.layer_types[layer] != qwen3_5_layer_type::linear_attention)
            continue;
        const auto* convolution =
            file.find(layer_prefix(prefix, layer) + "linear_attn.conv1d.weight");
        // MLX swaps the singleton convolution axis and folds +1 into RMS norms.
        if (convolution && config.linear_conv_kernel_dimension > 1)
            converted = convolution->shape
                == std::vector<std::size_t> { 2 * config.linear_key_width()
                                                  + config.linear_value_width(),
                                              config.linear_conv_kernel_dimension, 1 };
        break;
    }
    return { prefix, converted };
}

std::size_t
stored_tensor_count(const safetensors_file& file,
                    std::string_view prefix,
                    const weight_layout& layout)
{
    std::size_t count = 0;
    for (const auto& group : layout)
        for (const auto& tensor : group.tensors)
            count += file.find(std::string(prefix) + tensor.name)->type == safetensors_dtype::u32
                ? 3
                : 1;
    return count;
}
} // namespace

result<void, weight_error>
validate_qwen_weights(const safetensors_file& file, const qwen3_config& config)
{
    constexpr std::size_t max_per_layer = 25;
    if (config.layer_count > (std::numeric_limits<std::size_t>::max() - 7) / max_per_layer)
        return fail(weight_errc::tensor_count_overflow);
    CL_TRY(validate_weights(file, "model.", globals(config.vocabulary_size, config.hidden_size)));
    const bool has_output = file.find("lm_head.weight");
    if (has_output || !config.tie_word_embeddings)
        CL_TRY(
            validate_weights(file, "", output_layout(config.vocabulary_size, config.hidden_size)));
    const auto common = common_layer(config.hidden_size, config.intermediate_size);
    const auto attention = full_attention(config.hidden_size, config.query_width(),
                                          config.kv_width(), config.head_dimension, false);
    auto count =
        stored_tensor_count(file, "model.", globals(config.vocabulary_size, config.hidden_size));
    if (has_output)
        count += stored_tensor_count(file, "",
                                     output_layout(config.vocabulary_size, config.hidden_size));
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        auto prefix = layer_prefix("model.", layer);
        CL_TRY(validate_weights(file, prefix, common));
        CL_TRY(validate_weights(file, prefix, attention));
        count += stored_tensor_count(file, prefix, common)
            + stored_tensor_count(file, prefix, attention);
    }
    if (file.tensor_count() != count)
        return fail(weight_errc::unexpected_tensor_count);
    return {};
}

result<qwen_weights, weight_error>
load_qwen_weights(const metal_context& context,
                  const safetensors_file& file,
                  const qwen3_config& config,
                  weight_quantization quantization)
{
    CL_TRY(validate_qwen_weights(file, config));
    auto global = read_weights(context, file, "model.",
                               globals(config.vocabulary_size, config.hidden_size), quantization);
    if (!global)
        return fail(global.error());
    std::optional<matrix_weight> output;
    if (file.find("lm_head.weight")) {
        auto loaded =
            read_weights(context, file, "",
                         output_layout(config.vocabulary_size, config.hidden_size), quantization);
        if (!loaded)
            return fail(loaded.error());
        output.emplace(loaded->take_matrix("output"));
    }
    std::vector<qwen_layer_weights> layers;
    layers.reserve(config.layer_count);
    const auto common = common_layer(config.hidden_size, config.intermediate_size);
    const auto attention = full_attention(config.hidden_size, config.query_width(),
                                          config.kv_width(), config.head_dimension, false);
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        auto prefix = layer_prefix("model.", layer);
        auto base = read_weights(context, file, prefix, common, quantization);
        if (!base)
            return fail(base.error());
        auto mixer = read_weights(context, file, prefix, attention, quantization);
        if (!mixer)
            return fail(mixer.error());
        layers.push_back({ base->take_tensor("input_norm"), base->take_tensor("post_norm"),
                           mixer->take_tensor("query_norm"), mixer->take_tensor("key_norm"),
                           mixer->take_matrix("qkv"), mixer->take_matrix("output"),
                           base->take_matrix("gateup"), base->take_matrix("down") });
    }
    return qwen_weights { global->take_matrix("embedding"), global->take_tensor("norm"),
                          std::move(output), std::move(layers) };
}

result<void, weight_error>
validate_qwen3_5_weights(const safetensors_file& file, const qwen3_5_config& config)
{
    if (config.layer_types.size() != config.layer_count
        || !config.tie_word_embeddings
        || !config.attention_output_gate)
        return fail(weight_errc::invalid_configuration);
    const auto source = qwen3_5_layout(file, config);
    CL_TRY(
        validate_weights(file, source.prefix, globals(config.vocabulary_size, config.hidden_size)));
    const auto common = common_layer(config.hidden_size, config.intermediate_size);
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        auto prefix = layer_prefix(source.prefix, layer);
        CL_TRY(validate_weights(file, prefix, common));
        CL_TRY(validate_weights(file, prefix, mixer_layout(config, layer, source.converted)));
    }
    return {};
}

result<qwen3_5_weights, weight_error>
load_qwen3_5_weights(const metal_context& context,
                     const safetensors_file& file,
                     const qwen3_5_config& config,
                     weight_quantization quantization)
{
    CL_TRY(validate_qwen3_5_weights(file, config));
    const auto source = qwen3_5_layout(file, config);
    auto global = read_weights(context, file, source.prefix,
                               globals(config.vocabulary_size, config.hidden_size), quantization);
    if (!global)
        return fail(global.error());
    std::vector<qwen3_5_layer_weights> layers;
    layers.reserve(config.layer_count);
    const auto common = common_layer(config.hidden_size, config.intermediate_size);
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        auto prefix = layer_prefix(source.prefix, layer);
        auto base = read_weights(context, file, prefix, common, quantization);
        if (!base)
            return fail(base.error());
        auto mixer = read_weights(context, file, prefix,
                                  mixer_layout(config, layer, source.converted), quantization);
        if (!mixer)
            return fail(mixer.error());
        qwen3_5_mixer_weights weights = [&]() -> qwen3_5_mixer_weights {
            if (config.layer_types[layer] == qwen3_5_layer_type::full_attention)
                return qwen3_5_full_attention_weights { mixer->take_tensor("query_norm"),
                                                        mixer->take_tensor("key_norm"),
                                                        mixer->take_matrix("qkv"),
                                                        mixer->take_matrix("output") };
            return qwen3_5_linear_attention_weights {
                mixer->take_matrix("qkv"),         mixer->take_matrix("gate"),
                mixer->take_matrix("decay"),       mixer->take_matrix("rate"),
                mixer->take_tensor("convolution"), mixer->take_tensor("decay_log"),
                mixer->take_tensor("rate_bias"),   mixer->take_tensor("norm"),
                mixer->take_matrix("output")
            };
        }();
        if (source.converted && config.layer_types[layer] == qwen3_5_layer_type::linear_attention) {
            auto& linear = std::get<qwen3_5_linear_attention_weights>(weights);
            // Swapping a singleton axis leaves the convolution bytes in the same order.
            auto convolution =
                metal_tensor::make(context, dtype::bf16,
                                   { 2 * config.linear_key_width() + config.linear_value_width(), 1,
                                     config.linear_conv_kernel_dimension });
            if (!convolution)
                return fail(weight_errc::metal_allocation_failed, convolution.error(),
                            "convolution");
            std::memcpy(convolution->buffer().bytes().data(),
                        linear.convolution.buffer().bytes().data(),
                        linear.convolution.buffer().size_bytes());
            linear.convolution = std::move(*convolution);
            auto norm =
                metal_tensor::make(context, dtype::f32, { config.linear_value_head_dimension });
            if (!norm)
                return fail(weight_errc::metal_allocation_failed, norm.error(), "linear norm");
            for (std::size_t i = 0; i < config.linear_value_head_dimension; ++i) {
                std::uint16_t bits;
                std::memcpy(&bits, linear.norm.buffer().bytes().data() + i * sizeof(bits),
                            sizeof(bits));
                const auto value = bf16::from_bits(bits).to_float();
                std::memcpy(norm->buffer().bytes().data() + i * sizeof(value), &value,
                            sizeof(value));
            }
            linear.norm = std::move(*norm);
        }
        layers.push_back({ base->take_tensor("input_norm"), base->take_tensor("post_norm"),
                           base->take_matrix("gateup"), base->take_matrix("down"),
                           std::move(weights), !source.converted });
    }
    return qwen3_5_weights { global->take_matrix("embedding"), global->take_tensor("norm"),
                             std::move(layers), !source.converted };
}
} // namespace chibillm
