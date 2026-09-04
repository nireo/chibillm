#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "inference_engine.h"
#include "metal/metal_context.h"
#include "metal/metal_kv_cache.h"
#include "model_format/safetensors.h"
#include "qwen/qwen_configs.h"
#include "qwen/qwen_embedding.h"
#include "qwen/qwen_layer.h"
#include "qwen/qwen_model_runner.h"
#include "qwen/qwen_output.h"
#include "qwen/qwen_weights.h"
#include "metal_test_support.h"
#include "safetensors_test_support.h"
#include "tensor/bf16.h"

using chibillm::bf16;
using chibillm::embed_qwen_tokens;
using chibillm::encode_qwen_greedy;
using chibillm::load_qwen3_5_config;
using chibillm::load_qwen3_5_weights;
using chibillm::load_qwen_weights;
using chibillm::metal_context;
using chibillm::metal_kv_cache;
using chibillm::metal_tensor;
using chibillm::qwen3_5_config;
using chibillm::qwen3_5_full_attention_weights;
using chibillm::qwen3_5_layer_type;
using chibillm::qwen3_5_linear_attention_weights;
using chibillm::qwen3_config;
using chibillm::qwen_attention_metadata;
using chibillm::qwen_model_runner;
using chibillm::qwen_weights_errc;
using chibillm::read_qwen_greedy;
using chibillm::run_qwen_layers;
using chibillm::safetensors_file;
using chibillm::validate_qwen3_5_weights;
using chibillm::validate_qwen_weights;
using safetensors_test::temporary_file;
using namespace metal_test;

namespace {

struct tensor_spec {
    std::string name;
    std::string dtype;
    std::vector<std::size_t> shape;
};

qwen3_config
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

qwen3_5_config
qwen3_5_test_config()
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
        .partial_rotary_factor = 1.0F,
        .eos_token_id = 2,
        .tie_word_embeddings = true,
        .attention_output_gate = true,
        .full_attention_interval = 2,
        .layer_types = { qwen3_5_layer_type::linear_attention, qwen3_5_layer_type::full_attention },
        .linear_conv_kernel_dimension = 3,
        .linear_key_head_dimension = 2,
        .linear_key_head_count = 1,
        .linear_value_head_dimension = 2,
        .linear_value_head_count = 1,
        .mrope_interleaved = true,
        .mrope_sections = { 1, 0, 0 },
    };
}

std::vector<tensor_spec>
expected_tensors(const qwen3_config& config)
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

std::vector<tensor_spec>
expected_qwen3_5_tensors(const qwen3_5_config& config)
{
    std::vector<tensor_spec> tensors {
        { "model.language_model.embed_tokens.weight",
          "BF16",
          { config.vocabulary_size, config.hidden_size } },
        { "model.language_model.norm.weight", "BF16", { config.hidden_size } },
    };

    const auto add = [&](std::size_t layer, std::string suffix, std::string dtype,
                         std::vector<std::size_t> shape) {
        tensors.push_back(
            { "model.language_model.layers." + std::to_string(layer) + "." + std::move(suffix),
              std::move(dtype), std::move(shape) });
    };
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        add(layer, "input_layernorm.weight", "BF16", { config.hidden_size });
        add(layer, "post_attention_layernorm.weight", "BF16", { config.hidden_size });
        add(layer, "mlp.gate_proj.weight", "BF16",
            { config.intermediate_size, config.hidden_size });
        add(layer, "mlp.up_proj.weight", "BF16", { config.intermediate_size, config.hidden_size });
        add(layer, "mlp.down_proj.weight", "BF16",
            { config.hidden_size, config.intermediate_size });

        if (config.layer_types[layer] == qwen3_5_layer_type::full_attention) {
            add(layer, "self_attn.q_norm.weight", "BF16", { config.head_dimension });
            add(layer, "self_attn.k_norm.weight", "BF16", { config.head_dimension });
            add(layer, "self_attn.q_proj.weight", "BF16",
                { 2 * config.query_width(), config.hidden_size });
            add(layer, "self_attn.k_proj.weight", "BF16",
                { config.kv_width(), config.hidden_size });
            add(layer, "self_attn.v_proj.weight", "BF16",
                { config.kv_width(), config.hidden_size });
            add(layer, "self_attn.o_proj.weight", "BF16",
                { config.hidden_size, config.query_width() });
            continue;
        }

        const auto key_width = config.linear_key_width();
        const auto value_width = config.linear_value_width();
        const auto qkv_width = 2 * key_width + value_width;
        add(layer, "linear_attn.in_proj_qkv.weight", "BF16", { qkv_width, config.hidden_size });
        add(layer, "linear_attn.in_proj_z.weight", "BF16", { value_width, config.hidden_size });
        add(layer, "linear_attn.in_proj_a.weight", "BF16",
            { config.linear_value_head_count, config.hidden_size });
        add(layer, "linear_attn.in_proj_b.weight", "BF16",
            { config.linear_value_head_count, config.hidden_size });
        add(layer, "linear_attn.conv1d.weight", "BF16",
            { qkv_width, 1, config.linear_conv_kernel_dimension });
        add(layer, "linear_attn.A_log", "F32", { config.linear_value_head_count });
        add(layer, "linear_attn.dt_bias", "BF16", { config.linear_value_head_count });
        add(layer, "linear_attn.norm.weight", "F32", { config.linear_value_head_dimension });
        add(layer, "linear_attn.out_proj.weight", "BF16", { config.hidden_size, value_width });
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

class temporary_model_directory {
public:
    temporary_model_directory()
        : path_(std::filesystem::temp_directory_path() / "chibillm_qwen_runner_model")
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
        REQUIRE(std::filesystem::create_directory(path_));
    }

    ~temporary_model_directory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path&
    path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void
write_config(const std::filesystem::path& path, const qwen3_config& config)
{
    const nlohmann::json json {
        { "model_type", "qwen3" },
        { "hidden_act", "silu" },
        { "torch_dtype", "bfloat16" },
        { "attention_bias", false },
        { "use_sliding_window", false },
        { "sliding_window", nullptr },
        { "rope_scaling", nullptr },
        { "vocab_size", config.vocabulary_size },
        { "hidden_size", config.hidden_size },
        { "intermediate_size", config.intermediate_size },
        { "num_hidden_layers", config.layer_count },
        { "num_attention_heads", config.query_head_count },
        { "num_key_value_heads", config.kv_head_count },
        { "head_dim", config.head_dimension },
        { "max_position_embeddings", config.max_position_embeddings },
        { "rms_norm_eps", config.rms_epsilon },
        { "rope_theta", config.rope_theta },
        { "bos_token_id", config.bos_token_id },
        { "eos_token_id", config.eos_token_id },
        { "tie_word_embeddings", config.tie_word_embeddings },
    };
    std::ofstream output(path);
    REQUIRE(output.good());
    output << json;
    REQUIRE(output.good());
}

void
write_tokenizer(const std::filesystem::path& path)
{
    std::ofstream vocabulary(path / "vocab.json");
    REQUIRE(vocabulary.good());
    vocabulary << R"({"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7})";

    std::ofstream config(path / "tokenizer_config.json");
    REQUIRE(config.good());
    config << R"({"added_tokens_decoder":{}})";

    std::ofstream merges(path / "merges.txt");
    REQUIRE(merges.good());
    merges << "#version: 0.2\n";
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
    auto check_err = [&](std::vector<tensor_spec> tensors, const char* name, qwen_weights_errc err) {
        auto file = write_weights(std::move(tensors), name);
        auto weights = safetensors_file::open(file.path());
        REQUIRE(weights.has_value());
        CHECK(validate_qwen_weights(*weights, config).error() == err);
    };

    auto missing = expected_tensors(config);
    missing.erase(missing.begin() + 4);
    check_err(std::move(missing), "qwen_missing.safetensors", qwen_weights_errc::missing_tensor);

    auto wrong_dtype = expected_tensors(config);
    wrong_dtype[4].dtype = "F32";
    check_err(std::move(wrong_dtype), "qwen_dtype.safetensors", qwen_weights_errc::unsupported_dtype);

    auto wrong_shape = expected_tensors(config);
    wrong_shape[4].shape = { config.head_dimension + 1 };
    check_err(std::move(wrong_shape), "qwen_shape.safetensors", qwen_weights_errc::tensor_shape_mismatch);

    auto extra = expected_tensors(config);
    extra.push_back({ "unused.weight", "BF16", { 1 } });
    check_err(std::move(extra), "qwen_extra.safetensors", qwen_weights_errc::unexpected_tensor_count);
}

TEST_CASE("Qwen3.5 weight validation accepts its hybrid tensor layouts")
{
    const auto config = qwen3_5_test_config();
    auto file = write_weights(expected_qwen3_5_tensors(config),
                              "chibillm_qwen3_5_weights_valid.safetensors");
    auto weights = safetensors_file::open(file.path());
    REQUIRE(weights.has_value());
    CHECK(validate_qwen3_5_weights(*weights, config).has_value());
}

TEST_CASE("Qwen3.5 weight validation distinguishes architecture and storage types")
{
    const auto config = qwen3_5_test_config();
    auto check_err = [&](std::vector<tensor_spec> tensors, const char* name, qwen_weights_errc err) {
        auto file = write_weights(std::move(tensors), name);
        auto weights = safetensors_file::open(file.path());
        REQUIRE(weights.has_value());
        CHECK(validate_qwen3_5_weights(*weights, config).error() == err);
    };

    auto missing = expected_qwen3_5_tensors(config);
    const auto found_log = std::ranges::find_if(
        missing, [](const auto& t) { return t.name.ends_with("linear_attn.A_log"); });
    REQUIRE(found_log != missing.end());
    missing.erase(found_log);
    check_err(std::move(missing), "qwen3_5_missing.safetensors", qwen_weights_errc::missing_tensor);

    auto wrong_dtype = expected_qwen3_5_tensors(config);
    const auto found_dt = std::ranges::find_if(
        wrong_dtype, [](const auto& t) { return t.name.ends_with("linear_attn.A_log"); });
    REQUIRE(found_dt != wrong_dtype.end());
    found_dt->dtype = "BF16";
    check_err(std::move(wrong_dtype), "qwen3_5_dtype.safetensors", qwen_weights_errc::unsupported_dtype);

    check_err(expected_tensors(test_config()), "qwen3_as_3_5.safetensors", qwen_weights_errc::missing_tensor);
}

TEST_CASE("Qwen3.5 weights load linear and full-attention layers separately")
{
    const auto config = qwen3_5_test_config();
    auto file = write_weights(expected_qwen3_5_tensors(config),
                              "chibillm_qwen3_5_weights_load.safetensors");
    auto safetensors = safetensors_file::open(file.path());
    REQUIRE(safetensors.has_value());
    const auto* context = &test_context();

    auto weights = load_qwen3_5_weights(*context, *safetensors, config);
    REQUIRE(weights.has_value());
    REQUIRE(weights->layers.size() == config.layer_count);
    REQUIRE(std::holds_alternative<qwen3_5_linear_attention_weights>(weights->layers[0].mixer));
    REQUIRE(std::holds_alternative<qwen3_5_full_attention_weights>(weights->layers[1].mixer));

    const auto& linear = std::get<qwen3_5_linear_attention_weights>(weights->layers[0].mixer);
    CHECK(linear.decay_log.descriptor().type() == chibillm::dtype::f32);
    CHECK(linear.norm.descriptor().type() == chibillm::dtype::f32);
    CHECK(std::ranges::equal(linear.qkv_projection.descriptor().shape().dimensions(),
                             std::vector<std::size_t> { 6, config.hidden_size }));

    const auto& full = std::get<qwen3_5_full_attention_weights>(weights->layers[1].mixer);
    CHECK(std::ranges::equal(full.qkv_packed.descriptor().shape().dimensions(),
                             std::vector<std::size_t> { 12, config.hidden_size }));
}

TEST_CASE("Qwen3.5 official 0.8B checkpoint matches the weight schema")
{
    const std::filesystem::path model_directory { QWEN3_5_MODEL_PATH };
    const auto shard = model_directory / "model.safetensors-00001-of-00001.safetensors";
    if (!std::filesystem::exists(shard)) {
        MESSAGE("Qwen3.5 model is not installed; skipping local checkpoint validation");
        return;
    }

    auto config = load_qwen3_5_config(model_directory / "config.json");
    auto weights = safetensors_file::open(shard);
    REQUIRE(config.has_value());
    REQUIRE(weights.has_value());
    CHECK(validate_qwen3_5_weights(*weights, *config).has_value());
}

TEST_CASE("Qwen weights are loaded into resident Metal tensors")
{
    const auto config = test_config();
    auto file = write_weights(expected_tensors(config), "chibillm_qwen_weights_load.safetensors");
    auto safetensors = safetensors_file::open(file.path());
    REQUIRE(safetensors.has_value());
    const auto* context = &test_context();

    auto weights = load_qwen_weights(*context, *safetensors, config);
    REQUIRE(weights.has_value());
    CHECK(weights->layers.size() == config.layer_count);
    const std::vector<std::size_t> embedding_shape { config.vocabulary_size, config.hidden_size };
    CHECK(std::ranges::equal(weights->token_embedding.descriptor().shape().dimensions(),
                             embedding_shape));
    const std::vector<std::size_t> qkv_shape { config.query_width() + 2 * config.kv_width(),
                                               config.hidden_size };
    CHECK(std::ranges::equal(weights->layers[0].qkv_packed.descriptor().shape().dimensions(),
                             qkv_shape));

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
    const auto* context = &test_context();
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

TEST_CASE("Qwen output selects requested rows and samples their largest logits")
{
    const auto config = test_config();
    auto file = write_weights(expected_tensors(config), "chibillm_qwen_output.safetensors");
    auto safetensors = safetensors_file::open(file.path());
    REQUIRE(safetensors.has_value());
    const auto* context = &test_context();
    auto weights = load_qwen_weights(*context, *safetensors, config);
    REQUIRE(weights.has_value());

    std::vector<float> embeddings(config.vocabulary_size * config.hidden_size, 0.0F);
    embeddings[0] = 1.0F;
    embeddings[5] = 1.0F;
    embeddings[10] = 1.0F;
    write_bf16(weights->token_embedding, embeddings);
    write_bf16(weights->final_norm, std::vector<float>(config.hidden_size, 1.0F));

    std::vector<float> output(config.vocabulary_size * config.hidden_size, 0.0F);
    output[3 * config.hidden_size] = 2.0F;
    output[5 * config.hidden_size + 2] = 2.0F;
    write_bf16(weights->output, output);

    const std::vector<chibillm::token_id> input_tokens { 0, 1, 2 };
    auto hidden_states = embed_qwen_tokens(*context, *weights, input_tokens);
    REQUIRE(hidden_states.has_value());

    const std::vector<std::size_t> logits_indices { 2, 0 };
    auto sampled = encode_qwen_greedy(*context, config, *weights, *hidden_states, logits_indices);
    REQUIRE(sampled.has_value());
    CHECK(read_qwen_greedy(*sampled) == std::vector<chibillm::token_id> { 5, 3 });

    const std::vector<std::size_t> invalid_indices { input_tokens.size() };
    auto invalid = encode_qwen_greedy(*context, config, *weights, *hidden_states, invalid_indices);
    REQUIRE_FALSE(invalid.has_value());
    CHECK(invalid.error() == chibillm::qwen_output_errc::logits_index_out_of_range);
}

TEST_CASE("Qwen model runner executes a flattened multi-sequence batch")
{
    const auto config = test_config();
    temporary_model_directory model_directory;
    write_config(model_directory.path() / "config.json", config);
    write_tokenizer(model_directory.path());
    auto weights_file =
        write_weights(expected_tensors(config), "chibillm_qwen_runner_model/model.safetensors");

    auto runner = qwen_model_runner::make(model_directory.path(), load_shader_source(), 3, 2);
    REQUIRE(runner.has_value());
    CHECK(runner->config().vocabulary_size == config.vocabulary_size);

    auto engine = chibillm::inference_engine::make(
        {
            .max_sequences = 2,
            .max_batch_tokens = 4,
            .kv_block_count = 3,
            .kv_block_size = 2,
            .eos_token = config.eos_token_id,
        },
        *runner);
    auto first = chibillm::seq::make(
        10, { 1, 2, 3 }, { .temperature = 1.0F, .max_new_tokens = 2, .ignore_eos = false }, 2);
    auto second = chibillm::seq::make(
        20, { 4 }, { .temperature = 1.0F, .max_new_tokens = 2, .ignore_eos = false }, 2);
    REQUIRE(engine.has_value());
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(engine->add(std::move(*first)).has_value());
    REQUIRE(engine->add(std::move(*second)).has_value());
    while (!engine->is_finished()) {
        REQUIRE(engine->step().has_value());
    }
    const auto* first_result = engine->find_sequence(10);
    const auto* second_result = engine->find_sequence(20);
    REQUIRE(first_result != nullptr);
    REQUIRE(second_result != nullptr);
    CHECK(std::ranges::equal(first_result->completion_tokens(),
                             std::vector<chibillm::token_id> { 0, 0 }));
    CHECK(std::ranges::equal(second_result->completion_tokens(),
                             std::vector<chibillm::token_id> { 0, 0 }));

    const chibillm::model_batch invalid {
        .id = 2,
        .phase = chibillm::batch_phase::decode,
        .kv_block_size = 2,
        .tokens = { 1 },
        .positions = { 0 },
        .items = { { .id = 10,
                     .token_offset = 0,
                     .token_count = 1,
                     .logits_index = 0,
                     .block_table = { 3 } } },
    };
    auto rejected = runner->execute(invalid);
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error() == chibillm::model_runner_errc::inconsistent_batch);
}

TEST_CASE("Qwen layers execute forward pass with residual connections")
{
    const auto config = test_config();
    auto file = write_weights(expected_tensors(config), "chibillm_qwen_layers.safetensors");
    auto safetensors = safetensors_file::open(file.path());
    REQUIRE(safetensors.has_value());
    const auto* context = &test_context();
    auto weights = load_qwen_weights(*context, *safetensors, config);
    REQUIRE(weights.has_value());

    std::vector<float> embeddings(config.vocabulary_size * config.hidden_size, 0.0F);
    embeddings[4] = 1.0F;
    embeddings[5] = 2.0F;
    embeddings[6] = 3.0F;
    embeddings[7] = 4.0F;
    write_bf16(weights->token_embedding, embeddings);

    for (auto& layer : weights->layers) {
        write_bf16(layer.input_norm, std::vector<float>(config.hidden_size, 1.0F));
        write_bf16(layer.query_norm, std::vector<float>(config.head_dimension, 1.0F));
        write_bf16(layer.key_norm, std::vector<float>(config.head_dimension, 1.0F));
        const auto value_block =
            std::vector<float> { 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F };
        std::vector<float> packed_qkv(config.query_width() * config.hidden_size
                                          + 2 * config.kv_width() * config.hidden_size,
                                      0.0F);
        const auto value_offset = (config.query_width() + config.kv_width()) * config.hidden_size;
        std::copy(value_block.begin(), value_block.end(),
                  packed_qkv.begin() + static_cast<std::ptrdiff_t>(value_offset));
        write_bf16(layer.qkv_packed, packed_qkv);
        write_bf16(layer.attention_output,
                   { 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                     0.0F, 0.0F, 1.0F });
        write_bf16(layer.post_attention_norm, std::vector<float>(config.hidden_size, 0.0F));
    }

    auto cache_result = metal_kv_cache::make(*context,
                                             {
                                                 .layer_count = config.layer_count,
                                                 .block_count = 1,
                                                 .block_size = 1,
                                                 .kv_head_count = config.kv_head_count,
                                                 .head_dimension = config.head_dimension,
                                             });
    REQUIRE(cache_result.has_value());
    auto cache = std::move(*cache_result);

    const std::vector<chibillm::token_id> tokens { 1 };
    auto hidden_states = embed_qwen_tokens(*context, *weights, tokens);
    REQUIRE(hidden_states.has_value());

    const std::array<std::uint32_t, 1> zero { 0 };
    const std::array<std::uint32_t, 1> one { 1 };
    auto layer_output = run_qwen_layers(*context, config, *weights, std::move(*hidden_states),
                                        qwen_attention_metadata {
                                            .positions = zero,
                                            .slots = zero,
                                            .block_table = zero,
                                            .block_table_offsets = zero,
                                            .block_table_lengths = one,
                                        },
                                        cache);
    REQUIRE(layer_output.has_value());

    std::vector<float> layer_expected { 1.0F, 2.0F, 3.0F, 4.0F };
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        const auto rms = std::sqrt((layer_expected[0] * layer_expected[0]
                                    + layer_expected[1] * layer_expected[1]
                                    + layer_expected[2] * layer_expected[2]
                                    + layer_expected[3] * layer_expected[3])
                                       / 4.0F
                                   + config.rms_epsilon);
        const auto first = layer_expected[0] / rms;
        const auto second = layer_expected[1] / rms;
        layer_expected[0] += first;
        layer_expected[1] += second;
        layer_expected[2] += first;
        layer_expected[3] += second;
    }
    check_floats(*layer_output, layer_expected);
}
