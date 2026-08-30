#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

#include "result.h"

namespace chibillm {

enum class qwen_config_errc : std::uint8_t {
    file_read_failed,
    invalid_json,
    missing_field,
    invalid_field,
    unsupported_model_type,
    unsupported_configuration,
    invalid_geometry,
};

struct qwen3_config {
    std::size_t vocabulary_size;
    std::size_t hidden_size;
    std::size_t intermediate_size;
    std::size_t layer_count;
    std::size_t query_head_count;
    std::size_t kv_head_count;
    std::size_t head_dimension;
    std::size_t max_position_embeddings;
    float rms_epsilon;
    float rope_theta;
    std::int32_t bos_token_id;
    std::int32_t eos_token_id;
    bool tie_word_embeddings;

    [[nodiscard]] std::size_t query_width() const noexcept;
    [[nodiscard]] std::size_t kv_width() const noexcept;
    [[nodiscard]] std::size_t queries_per_kv_head() const noexcept;
};

enum class qwen3_5_layer_type : std::uint8_t {
    linear_attention,
    full_attention,
};

struct qwen3_5_config {
    std::size_t vocabulary_size;
    std::size_t hidden_size;
    std::size_t intermediate_size;
    std::size_t layer_count;
    std::size_t query_head_count;
    std::size_t kv_head_count;
    std::size_t head_dimension;
    std::size_t max_position_embeddings;
    float rms_epsilon;
    float rope_theta;
    float partial_rotary_factor;
    std::int32_t eos_token_id;
    bool tie_word_embeddings;
    bool attention_output_gate;

    std::size_t full_attention_interval;
    std::vector<qwen3_5_layer_type> layer_types;

    std::size_t linear_conv_kernel_dimension;
    std::size_t linear_key_head_dimension;
    std::size_t linear_key_head_count;
    std::size_t linear_value_head_dimension;
    std::size_t linear_value_head_count;

    bool mrope_interleaved;
    std::array<std::size_t, 3> mrope_sections;

    [[nodiscard]] std::size_t query_width() const noexcept;
    [[nodiscard]] std::size_t kv_width() const noexcept;
    [[nodiscard]] std::size_t queries_per_kv_head() const noexcept;
    [[nodiscard]] std::size_t rotary_dimension() const noexcept;
    [[nodiscard]] std::size_t linear_key_width() const noexcept;
    [[nodiscard]] std::size_t linear_value_width() const noexcept;
    [[nodiscard]] std::size_t full_attention_layer_count() const noexcept;
};

[[nodiscard]] result<qwen3_config, qwen_config_errc> parse_qwen3_config(std::string_view json);

[[nodiscard]] result<qwen3_config, qwen_config_errc>
load_qwen3_config(const std::filesystem::path& path);

[[nodiscard]] result<qwen3_5_config, qwen_config_errc> parse_qwen3_5_config(std::string_view json);

[[nodiscard]] result<qwen3_5_config, qwen_config_errc>
load_qwen3_5_config(const std::filesystem::path& path);

} // namespace chibillm
