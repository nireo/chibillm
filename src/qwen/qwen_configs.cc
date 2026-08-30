#include "qwen/qwen_configs.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace chibillm {
namespace {

using json = nlohmann::json;

result<const json*, qwen_config_errc>
required_member(const json& object, std::string_view key)
{
    const auto found = object.find(key);
    if (found == object.end()) {
        return fail(qwen_config_errc::missing_field);
    }
    return &*found;
}

result<json, qwen_config_errc>
parse_object(std::string_view json_text)
{
    auto object = json::parse(json_text, nullptr, false);
    if (object.is_discarded() || !object.is_object()) {
        return fail(qwen_config_errc::invalid_json);
    }
    return object;
}

result<json, qwen_config_errc>
load_object(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return fail(qwen_config_errc::file_read_failed);
    }
    const std::string json_text {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
    };
    if (input.bad()) {
        return fail(qwen_config_errc::file_read_failed);
    }
    return parse_object(json_text);
}

result<std::uint64_t, qwen_config_errc>
required_nonnegative_integer(const json& object, std::string_view key)
{
    const json* value = nullptr;
    CL_TRY_ASSIGN(value, required_member(object, key));
    if (value->is_number_unsigned()) {
        return value->get<std::uint64_t>();
    }
    if (value->is_number_integer()) {
        const auto integer = value->get<std::int64_t>();
        if (integer >= 0) {
            return static_cast<std::uint64_t>(integer);
        }
    }
    return fail(qwen_config_errc::invalid_field);
}

result<std::size_t, qwen_config_errc>
required_size(const json& object, std::string_view key)
{
    std::uint64_t value = 0;
    CL_TRY_ASSIGN(value, required_nonnegative_integer(object, key));
    if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
        return fail(qwen_config_errc::invalid_field);
    }
    return static_cast<std::size_t>(value);
}

result<std::int32_t, qwen_config_errc>
required_token_id(const json& object, std::string_view key)
{
    std::uint64_t value = 0;
    CL_TRY_ASSIGN(value, required_nonnegative_integer(object, key));
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        return fail(qwen_config_errc::invalid_field);
    }
    return static_cast<std::int32_t>(value);
}

result<float, qwen_config_errc>
required_positive_float(const json& object, std::string_view key)
{
    const json* member = nullptr;
    CL_TRY_ASSIGN(member, required_member(object, key));
    if (!member->is_number()) {
        return fail(qwen_config_errc::invalid_field);
    }
    const auto value = member->get<double>();
    if (!std::isfinite(value)
        || value <= 0.0
        || value > static_cast<double>(std::numeric_limits<float>::max())) {
        return fail(qwen_config_errc::invalid_field);
    }
    return static_cast<float>(value);
}

result<bool, qwen_config_errc>
required_bool(const json& object, std::string_view key)
{
    const json* value = nullptr;
    CL_TRY_ASSIGN(value, required_member(object, key));
    if (!value->is_boolean()) {
        return fail(qwen_config_errc::invalid_field);
    }
    return value->get<bool>();
}

result<std::string_view, qwen_config_errc>
required_string_view(const json& object, std::string_view key)
{
    const json* value = nullptr;
    CL_TRY_ASSIGN(value, required_member(object, key));
    if (!value->is_string()) {
        return fail(qwen_config_errc::invalid_field);
    }
    return value->get_ref<const std::string&>();
}

result<const json*, qwen_config_errc>
required_object(const json& object, std::string_view key)
{
    const json* value = nullptr;
    CL_TRY_ASSIGN(value, required_member(object, key));
    if (!value->is_object()) {
        return fail(qwen_config_errc::invalid_field);
    }
    return value;
}

result<void, qwen_config_errc>
require_string_value(const json& object,
                     std::string_view key,
                     std::string_view expected,
                     qwen_config_errc mismatch = qwen_config_errc::unsupported_configuration)
{
    std::string_view value;
    CL_TRY_ASSIGN(value, required_string_view(object, key));
    if (value != expected) {
        return fail(mismatch);
    }
    return {};
}

result<void, qwen_config_errc>
require_bool_value(const json& object,
                   std::string_view key,
                   bool expected,
                   qwen_config_errc mismatch = qwen_config_errc::unsupported_configuration)
{
    bool value = false;
    CL_TRY_ASSIGN(value, required_bool(object, key));
    if (value != expected) {
        return fail(mismatch);
    }
    return {};
}

result<void, qwen_config_errc>
require_null_or_missing(const json& object, std::string_view key)
{
    const auto found = object.find(key);
    if (found != object.end() && !found->is_null()) {
        return fail(qwen_config_errc::unsupported_configuration);
    }
    return {};
}

template <typename Config, typename ObjectParser>
result<Config, qwen_config_errc>
parse_config(std::string_view json_text, ObjectParser parse)
{
    auto object = parse_object(json_text);
    if (!object) {
        return fail(object.error());
    }
    return parse(*object);
}

template <typename Config, typename ObjectParser>
result<Config, qwen_config_errc>
load_config(const std::filesystem::path& path, ObjectParser parse)
{
    auto object = load_object(path);
    if (!object) {
        return fail(object.error());
    }
    return parse(*object);
}

result<std::vector<qwen3_5_layer_type>, qwen_config_errc>
parse_layer_types(const json& text_config)
{
    const json* values = nullptr;
    CL_TRY_ASSIGN(values, required_member(text_config, "layer_types"));
    if (!values->is_array()) {
        return fail(qwen_config_errc::invalid_field);
    }

    std::vector<qwen3_5_layer_type> types;
    types.reserve(values->size());
    for (const auto& value : *values) {
        if (!value.is_string()) {
            return fail(qwen_config_errc::invalid_field);
        }
        const auto name = value.get_ref<const std::string&>();
        if (name == "linear_attention") {
            types.push_back(qwen3_5_layer_type::linear_attention);
        } else if (name == "full_attention") {
            types.push_back(qwen3_5_layer_type::full_attention);
        } else {
            return fail(qwen_config_errc::unsupported_configuration);
        }
    }
    return types;
}

result<std::array<std::size_t, 3>, qwen_config_errc>
parse_mrope_sections(const json& rope)
{
    const json* values = nullptr;
    CL_TRY_ASSIGN(values, required_member(rope, "mrope_section"));
    if (!values->is_array() || values->size() != 3) {
        return fail(qwen_config_errc::invalid_field);
    }

    std::array<std::size_t, 3> sections {};
    for (std::size_t index = 0; index < sections.size(); ++index) {
        const auto& value = (*values)[index];
        if (!value.is_number_integer()) {
            return fail(qwen_config_errc::invalid_field);
        }
        const auto integer = value.get<std::int64_t>();
        if (integer <= 0
            || static_cast<std::uint64_t>(integer) > std::numeric_limits<std::size_t>::max()) {
            return fail(qwen_config_errc::invalid_field);
        }
        sections[index] = static_cast<std::size_t>(integer);
    }
    return sections;
}

result<qwen3_config, qwen_config_errc>
parse_qwen3_object(const json& object)
{
    CL_TRY(require_string_value(object, "model_type", "qwen3",
                                qwen_config_errc::unsupported_model_type));
    CL_TRY(require_string_value(object, "hidden_act", "silu"));
    CL_TRY(require_string_value(object, "torch_dtype", "bfloat16"));
    CL_TRY(require_bool_value(object, "attention_bias", false));
    CL_TRY(require_bool_value(object, "use_sliding_window", false));
    CL_TRY(require_null_or_missing(object, "sliding_window"));
    CL_TRY(require_null_or_missing(object, "rope_scaling"));

    qwen3_config config {};
    CL_TRY_ASSIGN(config.vocabulary_size, required_size(object, "vocab_size"));
    CL_TRY_ASSIGN(config.hidden_size, required_size(object, "hidden_size"));
    CL_TRY_ASSIGN(config.intermediate_size, required_size(object, "intermediate_size"));
    CL_TRY_ASSIGN(config.layer_count, required_size(object, "num_hidden_layers"));
    CL_TRY_ASSIGN(config.query_head_count, required_size(object, "num_attention_heads"));
    CL_TRY_ASSIGN(config.kv_head_count, required_size(object, "num_key_value_heads"));
    CL_TRY_ASSIGN(config.head_dimension, required_size(object, "head_dim"));
    CL_TRY_ASSIGN(config.max_position_embeddings, required_size(object, "max_position_embeddings"));
    CL_TRY_ASSIGN(config.rms_epsilon, required_positive_float(object, "rms_norm_eps"));
    CL_TRY_ASSIGN(config.rope_theta, required_positive_float(object, "rope_theta"));
    CL_TRY_ASSIGN(config.bos_token_id, required_token_id(object, "bos_token_id"));
    CL_TRY_ASSIGN(config.eos_token_id, required_token_id(object, "eos_token_id"));
    CL_TRY_ASSIGN(config.tie_word_embeddings, required_bool(object, "tie_word_embeddings"));

    const auto size_max = std::numeric_limits<std::size_t>::max();
    if (config.query_head_count % config.kv_head_count != 0
        || config.head_dimension % 2 != 0
        || config.query_head_count > size_max / config.head_dimension
        || config.kv_head_count > size_max / config.head_dimension
        || static_cast<std::size_t>(config.bos_token_id) >= config.vocabulary_size
        || static_cast<std::size_t>(config.eos_token_id) >= config.vocabulary_size) {
        return fail(qwen_config_errc::invalid_geometry);
    }
    return config;
}

result<qwen3_5_config, qwen_config_errc>
parse_qwen3_5_object(const json& object)
{
    CL_TRY(require_string_value(object, "model_type", "qwen3_5",
                                qwen_config_errc::unsupported_model_type));

    const json* text = nullptr;
    CL_TRY_ASSIGN(text, required_object(object, "text_config"));
    CL_TRY(require_string_value(*text, "model_type", "qwen3_5_text",
                                qwen_config_errc::unsupported_model_type));
    CL_TRY(require_string_value(*text, "hidden_act", "silu"));
    CL_TRY(require_string_value(*text, "dtype", "bfloat16"));
    CL_TRY(require_string_value(*text, "mamba_ssm_dtype", "float32"));
    CL_TRY(require_bool_value(*text, "attention_bias", false));
    CL_TRY(require_bool_value(*text, "attn_output_gate", true));

    const json* rope = nullptr;
    CL_TRY_ASSIGN(rope, required_object(*text, "rope_parameters"));
    CL_TRY(require_string_value(*rope, "rope_type", "default"));
    CL_TRY(require_bool_value(*rope, "mrope_interleaved", true));

    qwen3_5_config config {};
    CL_TRY_ASSIGN(config.vocabulary_size, required_size(*text, "vocab_size"));
    CL_TRY_ASSIGN(config.hidden_size, required_size(*text, "hidden_size"));
    CL_TRY_ASSIGN(config.intermediate_size, required_size(*text, "intermediate_size"));
    CL_TRY_ASSIGN(config.layer_count, required_size(*text, "num_hidden_layers"));
    CL_TRY_ASSIGN(config.query_head_count, required_size(*text, "num_attention_heads"));
    CL_TRY_ASSIGN(config.kv_head_count, required_size(*text, "num_key_value_heads"));
    CL_TRY_ASSIGN(config.head_dimension, required_size(*text, "head_dim"));
    CL_TRY_ASSIGN(config.max_position_embeddings, required_size(*text, "max_position_embeddings"));
    CL_TRY_ASSIGN(config.rms_epsilon, required_positive_float(*text, "rms_norm_eps"));
    CL_TRY_ASSIGN(config.eos_token_id, required_token_id(*text, "eos_token_id"));
    CL_TRY_ASSIGN(config.tie_word_embeddings, required_bool(*text, "tie_word_embeddings"));
    CL_TRY_ASSIGN(config.full_attention_interval, required_size(*text, "full_attention_interval"));
    CL_TRY_ASSIGN(config.linear_conv_kernel_dimension,
                  required_size(*text, "linear_conv_kernel_dim"));
    CL_TRY_ASSIGN(config.linear_key_head_dimension, required_size(*text, "linear_key_head_dim"));
    CL_TRY_ASSIGN(config.linear_key_head_count, required_size(*text, "linear_num_key_heads"));
    CL_TRY_ASSIGN(config.linear_value_head_dimension,
                  required_size(*text, "linear_value_head_dim"));
    CL_TRY_ASSIGN(config.linear_value_head_count, required_size(*text, "linear_num_value_heads"));
    CL_TRY_ASSIGN(config.rope_theta, required_positive_float(*rope, "rope_theta"));
    CL_TRY_ASSIGN(config.partial_rotary_factor,
                  required_positive_float(*rope, "partial_rotary_factor"));
    config.attention_output_gate = true;
    config.mrope_interleaved = true;
    CL_TRY_ASSIGN(config.mrope_sections, parse_mrope_sections(*rope));
    CL_TRY_ASSIGN(config.layer_types, parse_layer_types(*text));

    const auto size_max = std::numeric_limits<std::size_t>::max();
    const auto rotary_dimension_value =
        static_cast<double>(config.head_dimension) * config.partial_rotary_factor;
    const auto rounded_rotary_dimension = std::round(rotary_dimension_value);
    if (config.query_head_count % config.kv_head_count != 0
        || config.linear_value_head_count % config.linear_key_head_count != 0
        || config.query_head_count > size_max / config.head_dimension
        || config.kv_head_count > size_max / config.head_dimension
        || config.linear_key_head_count > size_max / config.linear_key_head_dimension
        || config.linear_value_head_count > size_max / config.linear_value_head_dimension
        || config.partial_rotary_factor > 1.0F
        || std::abs(rotary_dimension_value - rounded_rotary_dimension) > 1e-6
        || rounded_rotary_dimension <= 0.0
        || static_cast<std::size_t>(rounded_rotary_dimension) % 2 != 0
        || static_cast<std::size_t>(config.eos_token_id) >= config.vocabulary_size
        || config.layer_types.size() != config.layer_count) {
        return fail(qwen_config_errc::invalid_geometry);
    }

    const auto rotary_dimension = static_cast<std::size_t>(rounded_rotary_dimension);
    if (std::accumulate(config.mrope_sections.begin(), config.mrope_sections.end(),
                        std::size_t { 0 })
        != rotary_dimension / 2) {
        return fail(qwen_config_errc::invalid_geometry);
    }
    for (std::size_t layer = 0; layer < config.layer_types.size(); ++layer) {
        const auto expected = (layer + 1) % config.full_attention_interval == 0
            ? qwen3_5_layer_type::full_attention
            : qwen3_5_layer_type::linear_attention;
        if (config.layer_types[layer] != expected) {
            return fail(qwen_config_errc::invalid_geometry);
        }
    }
    return config;
}

} // namespace

std::size_t
qwen3_config::query_width() const noexcept
{
    return query_head_count * head_dimension;
}

std::size_t
qwen3_config::kv_width() const noexcept
{
    return kv_head_count * head_dimension;
}

std::size_t
qwen3_config::queries_per_kv_head() const noexcept
{
    return query_head_count / kv_head_count;
}

std::size_t
qwen3_5_config::query_width() const noexcept
{
    return query_head_count * head_dimension;
}

std::size_t
qwen3_5_config::kv_width() const noexcept
{
    return kv_head_count * head_dimension;
}

std::size_t
qwen3_5_config::queries_per_kv_head() const noexcept
{
    return query_head_count / kv_head_count;
}

std::size_t
qwen3_5_config::rotary_dimension() const noexcept
{
    return static_cast<std::size_t>(
        std::round(static_cast<double>(head_dimension) * partial_rotary_factor));
}

std::size_t
qwen3_5_config::linear_key_width() const noexcept
{
    return linear_key_head_count * linear_key_head_dimension;
}

std::size_t
qwen3_5_config::linear_value_width() const noexcept
{
    return linear_value_head_count * linear_value_head_dimension;
}

std::size_t
qwen3_5_config::full_attention_layer_count() const noexcept
{
    return static_cast<std::size_t>(
        std::count(layer_types.begin(), layer_types.end(), qwen3_5_layer_type::full_attention));
}

result<qwen3_config, qwen_config_errc>
parse_qwen3_config(std::string_view json_text)
{
    return parse_config<qwen3_config>(json_text, parse_qwen3_object);
}

result<qwen3_config, qwen_config_errc>
load_qwen3_config(const std::filesystem::path& path)
{
    return load_config<qwen3_config>(path, parse_qwen3_object);
}

result<qwen3_5_config, qwen_config_errc>
parse_qwen3_5_config(std::string_view json_text)
{
    return parse_config<qwen3_5_config>(json_text, parse_qwen3_5_object);
}

result<qwen3_5_config, qwen_config_errc>
load_qwen3_5_config(const std::filesystem::path& path)
{
    return load_config<qwen3_5_config>(path, parse_qwen3_5_object);
}

} // namespace chibillm
