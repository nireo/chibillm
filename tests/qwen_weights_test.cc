#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "metal/metal_context.h"
#include "model_format/safetensors.h"
#include "qwen/qwen_config.h"
#include "qwen/qwen_embedding.h"
#include "qwen/qwen_layer.h"
#include "qwen/qwen_weights.h"
#include "safetensors_test_support.h"
#include "tensor/bf16.h"

using chibillm::bf16;
using chibillm::embed_qwen_tokens;
using chibillm::load_qwen_weights;
using chibillm::metal_context;
using chibillm::metal_tensor;
using chibillm::normalize_qwen_qk;
using chibillm::project_qwen_qkv;
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
    std::byte value { 1 };
    for (auto& tensor : tensors) {
        const auto begin = data.size();
        safetensors_test::add_tensor(header, data, std::move(tensor.name), std::move(tensor.dtype),
                                     std::move(tensor.shape));
        std::fill(data.begin() + static_cast<std::ptrdiff_t>(begin), data.end(), value);
        value = static_cast<std::byte>(std::to_integer<unsigned int>(value) + 1);
    }
    return temporary_file(std::move(filename), header, data);
}

std::string
load_shader_source()
{
    std::ifstream input(CHIBILLM_SHADER_PATH);
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

void
write_bf16(metal_tensor& tensor, const std::vector<float>& values)
{
    std::vector<std::uint16_t> bits;
    bits.reserve(values.size());
    for (const auto value : values) {
        bits.push_back(bf16::from_float(value).bits());
    }
    REQUIRE(tensor.buffer().size_bytes() == bits.size() * sizeof(std::uint16_t));
    std::memcpy(tensor.buffer().bytes().data(), bits.data(), tensor.buffer().size_bytes());
}

void
check_floats(const metal_tensor& tensor, const std::vector<float>& expected)
{
    std::vector<float> values(tensor.descriptor().element_count());
    REQUIRE(values.size() == expected.size());
    std::memcpy(values.data(), tensor.buffer().bytes().data(), tensor.buffer().size_bytes());
    for (std::size_t index = 0; index < values.size(); ++index) {
        CHECK(values[index] == doctest::Approx(expected[index]));
    }
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

TEST_CASE("Qwen weights are loaded into resident Metal tensors")
{
    const auto config = test_config();
    auto file = write_weights(expected_tensors(config), "chibillm_qwen_weights_load.safetensors");
    auto safetensors = safetensors_file::open(file.path());
    REQUIRE(safetensors.has_value());
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto weights = load_qwen_weights(*context, *safetensors, config);
    REQUIRE(weights.has_value());
    CHECK(weights->layers.size() == config.layer_count);
    const std::vector<std::size_t> embedding_shape { config.vocabulary_size, config.hidden_size };
    CHECK(std::ranges::equal(weights->token_embedding.descriptor().shape().dimensions(),
                             embedding_shape));
    const std::vector<std::size_t> query_shape { config.query_width(), config.hidden_size };
    CHECK(std::ranges::equal(weights->layers[0].query.descriptor().shape().dimensions(),
                             query_shape));

    const auto embedding = weights->token_embedding.buffer().bytes();
    REQUIRE_FALSE(embedding.empty());
    CHECK(std::all_of(embedding.begin(), embedding.end(),
                      [](std::byte byte) { return byte == std::byte { 1 }; }));

    const auto first_layer_input_norm = weights->layers[0].input_norm.buffer().bytes();
    CHECK(std::all_of(first_layer_input_norm.begin(), first_layer_input_norm.end(),
                      [](std::byte byte) { return byte == std::byte { 4 }; }));
}

TEST_CASE("Qwen token embedding produces hidden-state rows")
{
    const auto config = test_config();
    auto file = write_weights(expected_tensors(config), "chibillm_qwen_embedding.safetensors");
    auto safetensors = safetensors_file::open(file.path());
    REQUIRE(safetensors.has_value());
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());
    auto weights = load_qwen_weights(*context, *safetensors, config);
    REQUIRE(weights.has_value());

    std::vector<std::uint16_t> embedding_bits;
    for (std::size_t token = 0; token < config.vocabulary_size; ++token) {
        for (std::size_t feature = 0; feature < config.hidden_size; ++feature) {
            embedding_bits.push_back(
                bf16::from_float(static_cast<float>(token * 10 + feature)).bits());
        }
    }
    std::memcpy(weights->token_embedding.buffer().bytes().data(), embedding_bits.data(),
                embedding_bits.size() * sizeof(std::uint16_t));

    const std::vector<chibillm::token_id> tokens { 2, 0 };
    auto hidden_states = embed_qwen_tokens(*context, *weights, tokens);
    REQUIRE(hidden_states.has_value());
    CHECK(std::ranges::equal(hidden_states->descriptor().shape().dimensions(),
                             std::vector<std::size_t> { 2, config.hidden_size }));

    std::vector<float> values(hidden_states->descriptor().element_count());
    std::memcpy(values.data(), hidden_states->buffer().bytes().data(),
                hidden_states->buffer().size_bytes());
    const std::vector<float> expected { 20.0F, 21.0F, 22.0F, 23.0F, 0.0F, 1.0F, 2.0F, 3.0F };
    CHECK(values == expected);
}

TEST_CASE("Qwen layer input produces normalized query key and value projections")
{
    const auto config = test_config();
    auto file = write_weights(expected_tensors(config), "chibillm_qwen_qkv.safetensors");
    auto safetensors = safetensors_file::open(file.path());
    REQUIRE(safetensors.has_value());
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());
    auto weights = load_qwen_weights(*context, *safetensors, config);
    REQUIRE(weights.has_value());

    std::vector<float> embeddings(config.vocabulary_size * config.hidden_size, 0.0F);
    embeddings[4] = 1.0F;
    embeddings[5] = 2.0F;
    embeddings[6] = 3.0F;
    embeddings[7] = 4.0F;
    write_bf16(weights->token_embedding, embeddings);
    write_bf16(weights->layers[0].input_norm, { 1.0F, 1.0F, 1.0F, 1.0F });
    write_bf16(weights->layers[0].query,
               { 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                 0.0F, 1.0F });
    write_bf16(weights->layers[0].key, { 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F });
    write_bf16(weights->layers[0].value, { 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F });
    write_bf16(weights->layers[0].query_norm, { 1.0F, 1.0F });
    write_bf16(weights->layers[0].key_norm, { 1.0F, 1.0F });

    const std::vector<chibillm::token_id> tokens { 1 };
    auto hidden_states = embed_qwen_tokens(*context, *weights, tokens);
    REQUIRE(hidden_states.has_value());
    auto qkv = project_qwen_qkv(*context, config, weights->layers[0], *hidden_states);
    REQUIRE(qkv.has_value());

    const auto scale = 1.0F / std::sqrt(7.5F + config.rms_epsilon);
    const std::vector<float> expected_query { scale, 2.0F * scale, 3.0F * scale, 4.0F * scale };
    const std::vector<float> expected_key { scale, 2.0F * scale };
    const std::vector<float> expected_value { 3.0F * scale, 4.0F * scale };
    check_floats(qkv->query, expected_query);
    check_floats(qkv->key, expected_key);
    check_floats(qkv->value, expected_value);

    auto normalized = normalize_qwen_qk(*context, config, weights->layers[0], std::move(*qkv));
    REQUIRE(normalized.has_value());
    const auto normalize_pair = [&](float first, float second) {
        const auto inverse_rms =
            1.0F / std::sqrt((first * first + second * second) / 2.0F + config.rms_epsilon);
        return std::vector<float> { first * inverse_rms, second * inverse_rms };
    };
    const auto first_query_head = normalize_pair(scale, 2.0F * scale);
    const auto second_query_head = normalize_pair(3.0F * scale, 4.0F * scale);
    check_floats(
        normalized->query,
        { first_query_head[0], first_query_head[1], second_query_head[0], second_query_head[1] });
    check_floats(normalized->key, first_query_head);
    check_floats(normalized->value, expected_value);
}
