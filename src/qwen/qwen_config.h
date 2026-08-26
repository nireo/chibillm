#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>

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

struct qwen_config {
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

[[nodiscard]] result<qwen_config, qwen_config_errc> parse_qwen_config(std::string_view json);

[[nodiscard]] result<qwen_config, qwen_config_errc>
load_qwen_config(const std::filesystem::path& path);

} // namespace chibillm
