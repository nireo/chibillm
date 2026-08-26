#include "qwen/qwen_config.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace chibillm {
namespace {

using json = nlohmann::json;
using config_result = result<qwen_config, qwen_config_errc>;

result<std::uint64_t, qwen_config_errc>
required_nonnegative_integer(const json& object, std::string_view key)
{
    const auto found = object.find(key);
    if (found == object.end()) {
        return fail(qwen_config_errc::missing_field);
    }

    std::uint64_t number = 0;
    if (found->is_number_unsigned()) {
        number = found->get<std::uint64_t>();
    } else if (found->is_number_integer()) {
        const auto signed_number = found->get<std::int64_t>();
        if (signed_number < 0) {
            return fail(qwen_config_errc::invalid_field);
        }
        number = static_cast<std::uint64_t>(signed_number);
    } else {
        return fail(qwen_config_errc::invalid_field);
    }

    return number;
}

result<std::size_t, qwen_config_errc>
required_size(const json& object, std::string_view key)
{
    auto number = required_nonnegative_integer(object, key);
    if (!number) {
        return fail(number.error());
    }
    if (*number == 0 || *number > std::numeric_limits<std::size_t>::max()) {
        return fail(qwen_config_errc::invalid_field);
    }
    return static_cast<std::size_t>(*number);
}

result<std::int32_t, qwen_config_errc>
required_token_id(const json& object, std::string_view key)
{
    auto number = required_nonnegative_integer(object, key);
    if (!number) {
        return fail(number.error());
    }
    if (*number > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        return fail(qwen_config_errc::invalid_field);
    }
    return static_cast<std::int32_t>(*number);
}

result<float, qwen_config_errc>
required_positive_float(const json& object, std::string_view key)
{
    const auto found = object.find(key);
    if (found == object.end()) {
        return fail(qwen_config_errc::missing_field);
    }
    if (!found->is_number()) {
        return fail(qwen_config_errc::invalid_field);
    }

    const auto number = found->get<double>();
    if (!std::isfinite(number)
        || number <= 0.0
        || number > static_cast<double>(std::numeric_limits<float>::max())) {
        return fail(qwen_config_errc::invalid_field);
    }
    return static_cast<float>(number);
}

result<bool, qwen_config_errc>
required_bool(const json& object, std::string_view key)
{
    const auto found = object.find(key);
    if (found == object.end()) {
        return fail(qwen_config_errc::missing_field);
    }
    if (!found->is_boolean()) {
        return fail(qwen_config_errc::invalid_field);
    }
    return found->get<bool>();
}

result<std::string, qwen_config_errc>
required_string(const json& object, std::string_view key)
{
    const auto found = object.find(key);
    if (found == object.end()) {
        return fail(qwen_config_errc::missing_field);
    }
    if (!found->is_string()) {
        return fail(qwen_config_errc::invalid_field);
    }
    return found->get<std::string>();
}

bool
is_null_or_missing(const json& object, std::string_view key)
{
    const auto found = object.find(key);
    return found == object.end() || found->is_null();
}

template <typename T>
bool
assign_or_fail(result<T, qwen_config_errc> value, T& destination, qwen_config_errc& error)
{
    if (!value) {
        error = value.error();
        return false;
    }
    destination = std::move(*value);
    return true;
}

config_result
parse_qwen_object(const json& object)
{
    auto model_type = required_string(object, "model_type");
    if (!model_type) {
        return fail(model_type.error());
    }
    if (*model_type != "qwen3") {
        return fail(qwen_config_errc::unsupported_model_type);
    }

    auto hidden_activation = required_string(object, "hidden_act");
    auto torch_dtype = required_string(object, "torch_dtype");
    auto attention_bias = required_bool(object, "attention_bias");
    auto use_sliding_window = required_bool(object, "use_sliding_window");
    if (!hidden_activation) {
        return fail(hidden_activation.error());
    }
    if (!torch_dtype) {
        return fail(torch_dtype.error());
    }
    if (!attention_bias) {
        return fail(attention_bias.error());
    }
    if (!use_sliding_window) {
        return fail(use_sliding_window.error());
    }
    if (*hidden_activation != "silu"
        || *torch_dtype != "bfloat16"
        || *attention_bias
        || *use_sliding_window
        || !is_null_or_missing(object, "sliding_window")
        || !is_null_or_missing(object, "rope_scaling")) {
        return fail(qwen_config_errc::unsupported_configuration);
    }

    qwen_config config {};
    qwen_config_errc error = qwen_config_errc::invalid_field;
    if (!assign_or_fail(required_size(object, "vocab_size"), config.vocabulary_size, error)
        || !assign_or_fail(required_size(object, "hidden_size"), config.hidden_size, error)
        || !assign_or_fail(required_size(object, "intermediate_size"), config.intermediate_size,
                           error)
        || !assign_or_fail(required_size(object, "num_hidden_layers"), config.layer_count, error)
        || !assign_or_fail(required_size(object, "num_attention_heads"), config.query_head_count,
                           error)
        || !assign_or_fail(required_size(object, "num_key_value_heads"), config.kv_head_count,
                           error)
        || !assign_or_fail(required_size(object, "head_dim"), config.head_dimension, error)
        || !assign_or_fail(required_size(object, "max_position_embeddings"),
                           config.max_position_embeddings, error)
        || !assign_or_fail(required_positive_float(object, "rms_norm_eps"), config.rms_epsilon,
                           error)
        || !assign_or_fail(required_positive_float(object, "rope_theta"), config.rope_theta, error)
        || !assign_or_fail(required_token_id(object, "bos_token_id"), config.bos_token_id, error)
        || !assign_or_fail(required_token_id(object, "eos_token_id"), config.eos_token_id, error)
        || !assign_or_fail(required_bool(object, "tie_word_embeddings"), config.tie_word_embeddings,
                           error)) {
        return fail(error);
    }

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

} // namespace

std::size_t
qwen_config::query_width() const noexcept
{
    return query_head_count * head_dimension;
}

std::size_t
qwen_config::kv_width() const noexcept
{
    return kv_head_count * head_dimension;
}

std::size_t
qwen_config::queries_per_kv_head() const noexcept
{
    return query_head_count / kv_head_count;
}

result<qwen_config, qwen_config_errc>
parse_qwen_config(std::string_view json_text)
{
    const auto object = json::parse(json_text, nullptr, false);
    if (object.is_discarded() || !object.is_object()) {
        return fail(qwen_config_errc::invalid_json);
    }
    return parse_qwen_object(object);
}

result<qwen_config, qwen_config_errc>
load_qwen_config(const std::filesystem::path& path)
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
    return parse_qwen_config(json_text);
}

} // namespace chibillm
