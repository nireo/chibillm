#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "model_format/safetensors.h"
#include "qwen/qwen_config.h"
#include "qwen/qwen_weights.h"
#include "safetensors_test_support.h"

using chibillm::qwen_config;
using chibillm::qwen_weights_errc;
using chibillm::safetensors_file;
using chibillm::validate_qwen_weights;
using safetensors_test::temporary_file;

namespace {

struct tensor_spec {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
};

qwen_config
test_config()
{
    return {
        .vocabulary_size = 8,
        .hidden_size = 4,
        .intermediate_size = 6,
        .layer_count = 2,
        .query_head_count = 2,
        .kv_head_count = 1,
        .head_dimension = 2,
        .max_position_embeddings = 32,
        .rms_epsilon = 1e-6F,
        .rope_theta = 10000.0F,
        .bos_token_id = 1,
        .eos_token_id = 2,
        .tie_word_embeddings = true,
    };
}

std::vector<tensor_spec>
expected_tensors(const qwen_config& config)
{
    std::vector<tensor_spec> tensors {
        { "model.embed_tokens.weight", "BF16", { config.vocabulary_size, config.hidden_size } },
        { "model.norm.weight", "BF16", { config.hidden_size } },
        { "lm_head.weight", "BF16", { config.vocabulary_size, config.hidden_size } },
    };

    const auto add = [&](std::size_t layer, std::string suffix, std::vector<std::size_t> shape) {
        tensors.push_back({ "model.layers." + std::to_string(layer) + "." + std::move(suffix),
                            "BF16", std::move(shape) });
    };
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        add(layer, "input_layernorm.weight", { config.hidden_size });
        add(layer, "post_attention_layernorm.weight", { config.hidden_size });
        add(layer, "self_attn.q_norm.weight", { config.head_dimension });
        add(layer, "self_attn.k_norm.weight", { config.head_dimension });
        add(layer, "self_attn.q_proj.weight", { config.query_width(), config.hidden_size });
        add(layer, "self_attn.k_proj.weight", { config.kv_width(), config.hidden_size });
        add(layer, "self_attn.v_proj.weight", { config.kv_width(), config.hidden_size });
        add(layer, "self_attn.o_proj.weight", { config.hidden_size, config.query_width() });
        add(layer, "mlp.gate_proj.weight", { config.intermediate_size, config.hidden_size });
        add(layer, "mlp.up_proj.weight", { config.intermediate_size, config.hidden_size });
        add(layer, "mlp.down_proj.weight", { config.hidden_size, config.intermediate_size });
    }
    return tensors;
}

temporary_file
write_weights(std::vector<tensor_spec> tensors, std::string filename)
{
    nlohmann::json header;
    std::vector<std::byte> data;
    for (auto& tensor : tensors) {
        safetensors_test::add_tensor(header, data, std::move(tensor.name), std::move(tensor.dtype),
                                     std::move(tensor.shape));
    }
    return temporary_file(std::move(filename), header, data);
}

} // namespace

TEST_CASE("Qwen weight validation accepts the expected tensor set")
{
    const auto config = test_config();
    auto file = write_weights(expected_tensors(config), "chibillm_qwen_weights_valid.safetensors");
    auto weights = safetensors_file::open(file.path());
    REQUIRE(weights.has_value());
    CHECK(validate_qwen_weights(*weights, config).has_value());
}

TEST_CASE("Qwen weight validation reports incompatible manifests")
{
    const auto config = test_config();

    SUBCASE("missing tensor")
    {
        auto tensors = expected_tensors(config);
        tensors.erase(tensors.begin() + 4);
        auto file = write_weights(std::move(tensors), "chibillm_qwen_weights_missing.safetensors");
        auto weights = safetensors_file::open(file.path());
        REQUIRE(weights.has_value());
        CHECK(validate_qwen_weights(*weights, config).error() == qwen_weights_errc::missing_tensor);
    }

    SUBCASE("wrong dtype")
    {
        auto tensors = expected_tensors(config);
        tensors[4].dtype = "F32";
        auto file = write_weights(std::move(tensors), "chibillm_qwen_weights_dtype.safetensors");
        auto weights = safetensors_file::open(file.path());
        REQUIRE(weights.has_value());
        CHECK(validate_qwen_weights(*weights, config).error()
              == qwen_weights_errc::unsupported_dtype);
    }

    SUBCASE("wrong shape")
    {
        auto tensors = expected_tensors(config);
        tensors[4].shape = { config.head_dimension + 1 };
        auto file = write_weights(std::move(tensors), "chibillm_qwen_weights_shape.safetensors");
        auto weights = safetensors_file::open(file.path());
        REQUIRE(weights.has_value());
        CHECK(validate_qwen_weights(*weights, config).error()
              == qwen_weights_errc::tensor_shape_mismatch);
    }

    SUBCASE("unexpected tensor")
    {
        auto tensors = expected_tensors(config);
        tensors.push_back({ "unused.weight", "BF16", { 1 } });
        auto file = write_weights(std::move(tensors), "chibillm_qwen_weights_extra.safetensors");
        auto weights = safetensors_file::open(file.path());
        REQUIRE(weights.has_value());
        CHECK(validate_qwen_weights(*weights, config).error()
              == qwen_weights_errc::unexpected_tensor_count);
    }
}
