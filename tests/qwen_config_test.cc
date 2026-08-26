#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <string>

#include "qwen/qwen_config.h"

using chibillm::load_qwen_config;
using chibillm::parse_qwen_config;
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

TEST_CASE("qwen config loads the Qwen3 checkpoint geometry")
{
    auto loaded = load_qwen_config(QWEN_CONFIG_FIXTURE_PATH);
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

TEST_CASE("qwen config rejects malformed and incomplete JSON")
{
    CHECK(parse_qwen_config("{").error() == qwen_config_errc::invalid_json);

    auto config = valid_config();
    replace_once(config, "\"vocab_size\": 32", "\"not_vocab_size\": 32");
    CHECK(parse_qwen_config(config).error() == qwen_config_errc::missing_field);
}

TEST_CASE("qwen config supports only the implemented Qwen architecture")
{
    SUBCASE("model type")
    {
        auto config = valid_config();
        replace_once(config, "\"qwen3\"", "\"llama\"");
        CHECK(parse_qwen_config(config).error() == qwen_config_errc::unsupported_model_type);
    }

    SUBCASE("activation")
    {
        auto config = valid_config();
        replace_once(config, "\"silu\"", "\"gelu\"");
        CHECK(parse_qwen_config(config).error() == qwen_config_errc::unsupported_configuration);
    }

    SUBCASE("dtype")
    {
        auto config = valid_config();
        replace_once(config, "\"bfloat16\"", "\"float16\"");
        CHECK(parse_qwen_config(config).error() == qwen_config_errc::unsupported_configuration);
    }

    SUBCASE("attention bias")
    {
        auto config = valid_config();
        replace_once(config, "\"attention_bias\": false", "\"attention_bias\": true");
        CHECK(parse_qwen_config(config).error() == qwen_config_errc::unsupported_configuration);
    }

    SUBCASE("sliding window")
    {
        auto config = valid_config();
        replace_once(config, "\"use_sliding_window\": false", "\"use_sliding_window\": true");
        CHECK(parse_qwen_config(config).error() == qwen_config_errc::unsupported_configuration);
    }
}

TEST_CASE("qwen config validates attention geometry and token ids")
{
    SUBCASE("grouped-query mapping")
    {
        auto config = valid_config();
        replace_once(config, "\"num_attention_heads\": 4", "\"num_attention_heads\": 3");
        CHECK(parse_qwen_config(config).error() == qwen_config_errc::invalid_geometry);
    }

    SUBCASE("RoPE dimension")
    {
        auto config = valid_config();
        replace_once(config, "\"head_dim\": 4", "\"head_dim\": 3");
        CHECK(parse_qwen_config(config).error() == qwen_config_errc::invalid_geometry);
    }

    SUBCASE("EOS token")
    {
        auto config = valid_config();
        replace_once(config, "\"eos_token_id\": 11", "\"eos_token_id\": 32");
        CHECK(parse_qwen_config(config).error() == qwen_config_errc::invalid_geometry);
    }
}

TEST_CASE("qwen config reports unreadable files")
{
    CHECK(load_qwen_config("this-file-does-not-exist.json").error()
          == qwen_config_errc::file_read_failed);
}
