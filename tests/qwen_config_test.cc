#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>

#include "qwen/qwen_configs.h"

using chibillm::load_qwen3_5_config;
using chibillm::load_qwen3_config;
using chibillm::parse_qwen3_5_config;
using chibillm::parse_qwen3_config;
using chibillm::qwen3_5_layer_type;
using chibillm::qwen_config_errc;

namespace {

std::string
valid_config()
{
    return R"({
        "attention_bias": false,
        "bos_token_id": 10,
        "eos_token_id": 11,
        "head_dim": 4,
        "hidden_act": "silu",
        "hidden_size": 8,
        "intermediate_size": 24,
        "max_position_embeddings": 128,
        "model_type": "qwen3",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_scaling": null,
        "rope_theta": 1000000,
        "sliding_window": null,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "use_sliding_window": false,
        "vocab_size": 32
    })";
}

void
replace_once(std::string& text, std::string_view from, std::string_view to)
{
    const auto position = text.find(from);
    REQUIRE(position != std::string::npos);
    text.replace(position, from.size(), to);
}

} // namespace

TEST_CASE("Qwen3 config loads the checkpoint geometry")
{
    auto loaded = load_qwen3_config(QWEN3_CONFIG_FIXTURE_PATH);
    REQUIRE(loaded.has_value());

    CHECK(loaded->vocabulary_size == 151936);
    CHECK(loaded->hidden_size == 1024);
    CHECK(loaded->intermediate_size == 3072);
    CHECK(loaded->layer_count == 28);
    CHECK(loaded->query_head_count == 16);
    CHECK(loaded->kv_head_count == 8);
    CHECK(loaded->head_dimension == 128);
    CHECK(loaded->max_position_embeddings == 40960);
    CHECK(loaded->rms_epsilon == doctest::Approx(1e-6F));
    CHECK(loaded->rope_theta == doctest::Approx(1000000.0F));
    CHECK(loaded->bos_token_id == 151643);
    CHECK(loaded->eos_token_id == 151645);
    CHECK(loaded->tie_word_embeddings);
    CHECK(loaded->query_width() == 2048);
    CHECK(loaded->kv_width() == 1024);
    CHECK(loaded->queries_per_kv_head() == 2);
}

TEST_CASE("Qwen3 config rejects malformed and incomplete JSON")
{
    CHECK(parse_qwen3_config("{").error() == qwen_config_errc::invalid_json);

    auto config = valid_config();
    replace_once(config, "\"vocab_size\": 32", "\"not_vocab_size\": 32");
    CHECK(parse_qwen3_config(config).error() == qwen_config_errc::missing_field);
}

TEST_CASE("Qwen3 config validates supported architecture and geometry")
{
    const std::vector<std::pair<std::pair<std::string_view, std::string_view>, qwen_config_errc>>
        cases {
            { { "\"qwen3\"", "\"llama\"" }, qwen_config_errc::unsupported_model_type },
            { { "\"silu\"", "\"gelu\"" }, qwen_config_errc::unsupported_configuration },
            { { "\"bfloat16\"", "\"float16\"" }, qwen_config_errc::unsupported_configuration },
            { { "\"attention_bias\": false", "\"attention_bias\": true" },
              qwen_config_errc::unsupported_configuration },
            { { "\"use_sliding_window\": false", "\"use_sliding_window\": true" },
              qwen_config_errc::unsupported_configuration },
            { { "\"num_attention_heads\": 4", "\"num_attention_heads\": 3" },
              qwen_config_errc::invalid_geometry },
            { { "\"head_dim\": 4", "\"head_dim\": 3" }, qwen_config_errc::invalid_geometry },
            { { "\"eos_token_id\": 11", "\"eos_token_id\": 32" },
              qwen_config_errc::invalid_geometry },
        };

    for (const auto& [replacement, expected_errc] : cases) {
        auto config = valid_config();
        replace_once(config, replacement.first, replacement.second);
        CHECK(parse_qwen3_config(config).error() == expected_errc);
    }
}

TEST_CASE("Qwen config loaders report unreadable files")
{
    CHECK(load_qwen3_config("this-file-does-not-exist.json").error()
          == qwen_config_errc::file_read_failed);
    CHECK(load_qwen3_5_config("this-file-does-not-exist.json").error()
          == qwen_config_errc::file_read_failed);
}

TEST_CASE("Qwen3.5 config loads the official 0.8B hybrid geometry")
{
    auto loaded = load_qwen3_5_config(QWEN3_5_CONFIG_FIXTURE_PATH);
    REQUIRE(loaded.has_value());

    CHECK(loaded->vocabulary_size == 248320);
    CHECK(loaded->hidden_size == 1024);
    CHECK(loaded->intermediate_size == 3584);
    CHECK(loaded->layer_count == 24);
    CHECK(loaded->query_head_count == 8);
    CHECK(loaded->kv_head_count == 2);
    CHECK(loaded->head_dimension == 256);
    CHECK(loaded->query_width() == 2048);
    CHECK(loaded->kv_width() == 512);
    CHECK(loaded->queries_per_kv_head() == 4);
    CHECK(loaded->rotary_dimension() == 64);
    CHECK(loaded->linear_key_width() == 2048);
    CHECK(loaded->linear_value_width() == 2048);
    CHECK(loaded->linear_conv_kernel_dimension == 4);
    CHECK(loaded->full_attention_interval == 4);
    CHECK(loaded->full_attention_layer_count() == 6);
    CHECK(loaded->layer_types[0] == qwen3_5_layer_type::linear_attention);
    CHECK(loaded->layer_types[3] == qwen3_5_layer_type::full_attention);
    CHECK(loaded->max_position_embeddings == 262144);
    CHECK(loaded->rope_theta == doctest::Approx(10000000.0F));
    CHECK(loaded->partial_rotary_factor == doctest::Approx(0.25F));
    CHECK(loaded->eos_token_id == 248044);
    CHECK(loaded->tie_word_embeddings);
    CHECK(loaded->attention_output_gate);
    CHECK(loaded->mrope_interleaved);
    CHECK(loaded->mrope_sections == std::array<std::size_t, 3> { 11, 11, 10 });
}

TEST_CASE("Qwen3 and Qwen3.5 parsers remain architecture-specific")
{
    std::ifstream input(QWEN3_5_CONFIG_FIXTURE_PATH);
    REQUIRE(input.good());
    const std::string qwen3_5_json { std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>() };

    CHECK(parse_qwen3_config(qwen3_5_json).error() == qwen_config_errc::unsupported_model_type);
    CHECK(parse_qwen3_5_config(valid_config()).error() == qwen_config_errc::unsupported_model_type);
}
