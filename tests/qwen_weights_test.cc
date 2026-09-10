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
#include "metal_test_support.h"
#include "model_factory.h"
#include "model_format/safetensors.h"
#include "qwen/qwen3_5_model_state.h"
#include "qwen/qwen_configs.h"
#include "qwen/qwen_layer.h"
#include "qwen/qwen_model_runner.h"
#include "qwen/qwen_weights.h"
#include "safetensors_test_support.h"
#include "tensor/bf16.h"
#include "tensor/embedding.h"
#include "tensor/output.h"

using chibillm::attention_metadata;
using chibillm::bf16;
using chibillm::embed_tokens;
using chibillm::encode_greedy;
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
using chibillm::qwen_model_runner;
using chibillm::read_greedy;
using chibillm::run_qwen_layers;
using chibillm::safetensors_file;
using chibillm::validate_qwen3_5_weights;
using chibillm::validate_qwen_weights;
using chibillm::weight_errc;
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

void
write_config(const std::filesystem::path& path, const qwen3_5_config& config)
{
    std::ifstream fixture(QWEN3_5_CONFIG_FIXTURE_PATH);
    auto json = nlohmann::json::parse(fixture);
    auto& text = json["text_config"];
    text.update({
        { "vocab_size", config.vocabulary_size },
        { "hidden_size", config.hidden_size },
        { "intermediate_size", config.intermediate_size },
        { "num_hidden_layers", config.layer_count },
        { "num_attention_heads", config.query_head_count },
        { "num_key_value_heads", config.kv_head_count },
        { "head_dim", config.head_dimension },
        { "max_position_embeddings", config.max_position_embeddings },
        { "rms_norm_eps", config.rms_epsilon },
        { "eos_token_id", config.eos_token_id },
        { "full_attention_interval", config.full_attention_interval },
        { "layer_types", { "linear_attention", "full_attention" } },
        { "linear_conv_kernel_dim", config.linear_conv_kernel_dimension },
        { "linear_key_head_dim", config.linear_key_head_dimension },
        { "linear_num_key_heads", config.linear_key_head_count },
        { "linear_value_head_dim", config.linear_value_head_dimension },
        { "linear_num_value_heads", config.linear_value_head_count },
    });
    text["rope_parameters"]["rope_theta"] = config.rope_theta;
    text["rope_parameters"]["partial_rotary_factor"] = config.partial_rotary_factor;
    text["rope_parameters"]["mrope_section"] = config.mrope_sections;
    std::ofstream(path) << json;
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
    auto check_err = [&](std::vector<tensor_spec> tensors, const char* name, weight_errc err) {
        auto file = write_weights(std::move(tensors), name);
        auto weights = safetensors_file::open(file.path());
        REQUIRE(weights.has_value());
        CHECK(validate_qwen_weights(*weights, config).error() == err);
    };

    auto missing = expected_tensors(config);
    missing.erase(missing.begin() + 4);
    check_err(std::move(missing), "qwen_missing.safetensors", weight_errc::missing_tensor);

    auto wrong_dtype = expected_tensors(config);
    wrong_dtype[4].dtype = "F32";
    check_err(std::move(wrong_dtype), "qwen_dtype.safetensors", weight_errc::unsupported_dtype);

    auto wrong_shape = expected_tensors(config);
    wrong_shape[4].shape = { config.head_dimension + 1 };
    check_err(std::move(wrong_shape), "qwen_shape.safetensors", weight_errc::tensor_shape_mismatch);

    auto extra = expected_tensors(config);
    extra.push_back({ "unused.weight", "BF16", { 1 } });
    check_err(std::move(extra), "qwen_extra.safetensors", weight_errc::unexpected_tensor_count);
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
    auto check_err = [&](std::vector<tensor_spec> tensors, const char* name, weight_errc err) {
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
    check_err(std::move(missing), "qwen3_5_missing.safetensors", weight_errc::missing_tensor);

    auto wrong_dtype = expected_qwen3_5_tensors(config);
    const auto found_dt = std::ranges::find_if(
        wrong_dtype, [](const auto& t) { return t.name.ends_with("linear_attn.A_log"); });
    REQUIRE(found_dt != wrong_dtype.end());
    found_dt->dtype = "BF16";
    check_err(std::move(wrong_dtype), "qwen3_5_dtype.safetensors", weight_errc::unsupported_dtype);

    check_err(expected_tensors(test_config()), "qwen3_as_3_5.safetensors",
              weight_errc::missing_tensor);
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
    auto hidden_states = embed_tokens(*context, weights->token_embedding, tokens);
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
    auto hidden_states = embed_tokens(*context, weights->token_embedding, input_tokens);
    REQUIRE(hidden_states.has_value());

    const std::vector<std::size_t> logits_indices { 2, 0 };
    auto sampled = encode_greedy(*context, weights->final_norm, weights->output, config.rms_epsilon,
                                 *hidden_states, logits_indices);
    REQUIRE(sampled.has_value());
    CHECK(read_greedy(*sampled) == std::vector<chibillm::token_id> { 5, 3 });

    const std::vector<std::size_t> invalid_indices { input_tokens.size() };
    auto invalid = encode_greedy(*context, weights->final_norm, weights->output, config.rms_epsilon,
                                 *hidden_states, invalid_indices);
    REQUIRE_FALSE(invalid.has_value());
    CHECK(invalid.error() == chibillm::tensor_op_errc::logits_index_out_of_range);
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
    auto first = chibillm::seq::make(10, { 1, 2, 3 }, { .max_new_tokens = 2, .ignore_eos = false });
    auto second = chibillm::seq::make(20, { 4 }, { .max_new_tokens = 2, .ignore_eos = false });
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
    auto state = runner->make_state({ .kv_block_count = 3, .kv_block_size = 2 });
    REQUIRE(state.has_value());
    auto rejected = runner->execute(invalid, **state);
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error() == chibillm::model_runner_errc::inconsistent_batch);
}

TEST_CASE("Qwen3.5 factory loads a sole shard and generates with tied zero-centered output")
{
    auto config = qwen3_5_test_config();
    config.head_dimension = 8;
    config.mrope_sections = { 2, 1, 1 };
    temporary_model_directory directory;
    write_config(directory.path() / "config.json", config);
    write_tokenizer(directory.path());
    nlohmann::json header;
    std::vector<std::byte> data;
    for (const auto& spec : expected_qwen3_5_tensors(config))
        safetensors_test::add_tensor(header, data, spec.name, spec.dtype, spec.shape);
    // Zero mixers preserve the embeddings. Token 4 wins the tied projection;
    // forgetting the +1 in the final RMSNorm instead yields token 0.
    for (std::size_t token = 0; token < config.vocabulary_size; ++token) {
        const auto value = bf16::from_float(token == 4 ? 2.0F : 1.0F).bits();
        std::memcpy(data.data() + token * config.hidden_size * sizeof(value), &value,
                    sizeof(value));
    }
    temporary_file weights((directory.path() / "model-00001-of-00001.safetensors").string(), header,
                           data);
    auto loaded = chibillm::load_model(directory.path(), load_shader_source(), 8, 2, "tiny-hybrid");
    REQUIRE(loaded.has_value());
    auto& runner = **loaded;
    REQUIRE(dynamic_cast<chibillm::qwen3_5_model_runner*>(&runner) != nullptr);
    CHECK(runner.info().id == "tiny-hybrid");
    CHECK(runner.info().max_context_tokens == 16);
    CHECK(runner.info().eos_token == config.eos_token_id);
    auto state = runner.make_state({ .kv_block_count = 8, .kv_block_size = 2 });
    REQUIRE(state.has_value());
    CHECK(dynamic_cast<chibillm::qwen3_5_model_state*>(state->get()) != nullptr);
    CHECK_FALSE(runner.make_state({ .kv_block_count = 9, .kv_block_size = 2 }).has_value());
    CHECK_FALSE(runner.make_state({ .kv_block_count = 8, .kv_block_size = 1 }).has_value());
    CHECK(runner.execute({}, **state).error() == chibillm::model_runner_errc::empty_batch);
    CHECK(runner.encode_chat({}).error() == chibillm::model_runner_errc::invalid_chat);

    auto engine = chibillm::inference_engine::make({ .max_sequences = 2,
                                                     .max_batch_tokens = 2,
                                                     .kv_block_count = 8,
                                                     .kv_block_size = 2,
                                                     .eos_token = config.eos_token_id },
                                                   runner);
    REQUIRE(engine.has_value());
    for (int repeat = 0; repeat < 2; ++repeat) {
        auto first = chibillm::seq::make(10, { 1, 2, 3, 1, 2 }, { .max_new_tokens = 3 });
        auto second = chibillm::seq::make(20, { 3 }, { .max_new_tokens = 3 });
        REQUIRE(first.has_value());
        REQUIRE(second.has_value());
        REQUIRE(engine->add(std::move(*first)).has_value());
        REQUIRE(engine->add(std::move(*second)).has_value());
        auto partial = engine->step();
        REQUIRE(partial.has_value());
        CHECK(partial->empty());
        for (int step = 0; !engine->is_finished() && step < 16; ++step)
            REQUIRE(engine->step().has_value());
        REQUIRE(engine->is_finished());
        for (const auto id : { 10, 20 }) {
            const auto* sequence = engine->find_sequence(id);
            REQUIRE(sequence != nullptr);
            CHECK(std::ranges::equal(sequence->completion_tokens(),
                                     std::vector<chibillm::token_id> { 4, 4, 4 }));
            CHECK(sequence->reason() == chibillm::finish_reason::len_limit);
            CHECK(runner.decode(sequence->completion_tokens()).value() == "eee");
            REQUIRE(engine->remove(id).has_value());
        }
    }
}

TEST_CASE("Qwen3.5 official checkpoint generates consistently across batching and cancellation")
{
    const std::filesystem::path directory { QWEN3_5_MODEL_PATH };
    if (!std::filesystem::exists(directory / "config.json")) {
        MESSAGE("Qwen3.5 model is not installed; skipping local generation validation");
        return;
    }
    auto loaded = chibillm::load_model(directory, load_shader_source(), 16, 16, "qwen3.5");
    REQUIRE(loaded.has_value());
    auto& runner = **loaded;
    const std::vector<chibillm::chat_message> first_chat {
        { "user", "Reply with just the answer: What is 2 + 2?" },
    };
    const std::vector<chibillm::chat_message> second_chat {
        { "user", "Reply with just the answer: What is 3 + 3?" },
    };
    auto first_prompt = runner.encode_chat(first_chat);
    auto second_prompt = runner.encode_chat(second_chat);
    REQUIRE(first_prompt.has_value());
    REQUIRE(second_prompt.has_value());
    std::vector<std::vector<chibillm::token_id>> reference;
    for (const std::size_t budget : { 128, 7 }) {
        auto engine = chibillm::inference_engine::make({ .max_sequences = 2,
                                                         .max_batch_tokens = budget,
                                                         .kv_block_count = 16,
                                                         .kv_block_size = 16,
                                                         .eos_token = runner.info().eos_token },
                                                       runner);
        REQUIRE(engine.has_value());
        const auto add = [&](chibillm::seq_id id, const std::vector<chibillm::token_id>& prompt) {
            auto sequence = chibillm::seq::make(id, prompt, { .max_new_tokens = 16 });
            REQUIRE(sequence.has_value());
            REQUIRE(engine->add(std::move(*sequence)).has_value());
        };
        if (budget == 7) {
            // Release a partially populated recurrent state and reuse its ID/pages.
            add(1, *second_prompt);
            auto partial = engine->step();
            REQUIRE(partial.has_value());
            CHECK(partial->empty());
            REQUIRE(engine->cancel(1).has_value());
            REQUIRE(engine->remove(1).has_value());
        }
        add(1, *first_prompt);
        add(2, *second_prompt);
        for (int step = 0; !engine->is_finished() && step < 64; ++step)
            REQUIRE(engine->step().has_value());
        REQUIRE(engine->is_finished());
        for (const auto id : { 1, 2 }) {
            const auto* sequence = engine->find_sequence(id);
            REQUIRE(sequence != nullptr);
            CHECK(sequence->reason() == chibillm::finish_reason::eos);
            const auto tokens = sequence->completion_tokens();
            auto decoded = runner.decode(tokens);
            REQUIRE(decoded.has_value());
            const auto start = decoded->find_first_not_of(" \r\n\t");
            const auto end = decoded->find_last_not_of(" \r\n\t");
            REQUIRE(start != std::string::npos);
            CHECK(decoded->substr(start, end - start + 1) == (id == 1 ? "4" : "6"));
            auto decoder = runner.make_decoder();
            std::string streamed;
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                auto delta = decoder->push(tokens[i], i + 1 == tokens.size());
                REQUIRE(delta.has_value());
                streamed += *delta;
            }
            CHECK(streamed == *decoded);
            if (budget == 128)
                reference.emplace_back(tokens.begin(), tokens.end());
            else
                CHECK(std::ranges::equal(tokens, reference[id - 1]));
        }
    }
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
    auto hidden_states = embed_tokens(*context, weights->token_embedding, tokens);
    REQUIRE(hidden_states.has_value());

    const std::array<std::uint32_t, 1> zero { 0 };
    const std::array<std::uint32_t, 1> one { 1 };
    auto layer_output = run_qwen_layers(*context, config, *weights, std::move(*hidden_states),
                                        attention_metadata {
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

TEST_CASE("Qwen3.5 full attention matches CPU for prefill and cached decode")
{
    auto config = qwen3_5_test_config();
    config.head_dimension = 4;
    config.partial_rotary_factor = 0.5F;
    config.rms_epsilon = 0.125F;

    auto file = write_weights(expected_qwen3_5_tensors(config), "qwen3_5_attention.safetensors");
    auto safetensors = safetensors_file::open(file.path());
    REQUIRE(safetensors.has_value());
    auto made = metal_context::make(load_shader_source());
    REQUIRE(made.has_value());
    auto& context = *made;
    auto weights = load_qwen3_5_weights(context, *safetensors, config);
    REQUIRE(weights.has_value());
    auto& layer = weights->layers[1];
    auto& attention = std::get<qwen3_5_full_attention_weights>(layer.mixer);

    const std::vector<float> input { 1, 2, -1, 0.5F, -2, 1, 3, 1, 0.5F, -1, 2, 3 };
    const std::vector<float> input_norm { 0, 0.5F, -0.25F, 0.125F };
    const std::vector<float> qk_norm { 0, 0.25F, -0.25F, 0.5F };
    std::vector<float> projection(24 * 4), output_weight(4 * 8);
    for (std::size_t i = 0; i < projection.size(); ++i)
        projection[i] = float(int((i * 7 + 3) % 13) - 6) * 0.125F;
    for (std::size_t i = 0; i < output_weight.size(); ++i)
        output_weight[i] = float(int((i * 3 + 1) % 7) - 3) * 0.125F;

    write_bf16(layer.input_norm, input_norm);
    write_bf16(attention.query_norm, qk_norm);
    write_bf16(attention.key_norm, qk_norm);
    write_bf16(attention.qkv_packed, projection);
    write_bf16(attention.output, output_weight);

    // Independent CPU reference: two query heads share one KV head.
    auto normalize = [&](std::vector<float> x, const std::vector<float>& weight) {
        float square = 0;
        for (float value : x)
            square += value * value;
        const float scale = 1.0F / std::sqrt(square / float(x.size()) + config.rms_epsilon);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] *= scale * (1.0F + weight[i]);
        return x;
    };
    auto rotate = [](std::vector<float> x, std::size_t position) {
        const float cosine = std::cos(float(position)), sine = std::sin(float(position));
        const float a = x[0], b = x[1];
        x[0] = a * cosine - b * sine;
        x[1] = b * cosine + a * sine;
        return x;
    };

    std::vector<std::vector<float>> keys, values, gates, queries;
    for (std::size_t row = 0; row < 3; ++row) {
        auto hidden = normalize(
            std::vector<float>(input.begin() + row * 4, input.begin() + (row + 1) * 4), input_norm);
        std::vector<float> projected(24, 0);
        for (std::size_t out = 0; out < 24; ++out)
            for (std::size_t in = 0; in < 4; ++in)
                projected[out] += projection[out * 4 + in] * hidden[in];

        for (std::size_t head = 0; head < 2; ++head) {
            auto start = projected.begin() + head * 8;
            queries.push_back(
                rotate(normalize(std::vector<float>(start, start + 4), qk_norm), row));
            gates.emplace_back(start + 4, start + 8);
        }
        keys.push_back(rotate(
            normalize(std::vector<float>(projected.begin() + 16, projected.begin() + 20), qk_norm),
            row));
        values.emplace_back(projected.begin() + 20, projected.end());
    }

    auto expected = input;
    for (std::size_t row = 0; row < 3; ++row) {
        std::vector<float> attended(8, 0);
        for (std::size_t head = 0; head < 2; ++head) {
            std::vector<float> scores(row + 1, 0);
            for (std::size_t token = 0; token <= row; ++token)
                for (std::size_t feature = 0; feature < 4; ++feature)
                    scores[token] += queries[row * 2 + head][feature] * keys[token][feature] / 2.0F;

            const float maximum = *std::max_element(scores.begin(), scores.end());
            float sum = 0;
            for (auto& score : scores) {
                score = std::exp(score - maximum);
                sum += score;
            }
            for (std::size_t feature = 0; feature < 4; ++feature) {
                float value = 0;
                for (std::size_t token = 0; token <= row; ++token)
                    value += scores[token] / sum * values[token][feature];
                attended[head * 4 + feature] =
                    value / (1.0F + std::exp(-gates[row * 2 + head][feature]));
            }
        }
        for (std::size_t out = 0; out < 4; ++out)
            for (std::size_t in = 0; in < 8; ++in)
                expected[row * 4 + out] += output_weight[out * 8 + in] * attended[in];
    }

    // Both paths target compact cache layer 1 and physical block 1.
    for (bool decode : { false, true }) {
        auto cache = metal_kv_cache::make(context, { 2, 2, 4, 1, 4 });
        REQUIRE(cache.has_value());
        write_floats(cache->keys(), std::vector<float>(cache->element_count(), 0));
        write_floats(cache->values(), std::vector<float>(cache->element_count(), 0));
        std::vector<float> actual;

        for (std::size_t start = 0; start < 3; start += decode ? 1 : 3) {
            const std::size_t count = decode ? 1 : 3;
            auto hidden = make_tensor(context, chibillm::dtype::f32, { count, 4 });
            write_floats(
                hidden,
                std::vector<float>(input.begin() + start * 4, input.begin() + (start + count) * 4));
            std::vector<std::uint32_t> positions(count), slots(count), table(count, 1),
                offsets(count), lengths(count, 1);
            for (std::size_t i = 0; i < count; ++i) {
                positions[i] = start + i;
                slots[i] = 4 + start + i;
                offsets[i] = i;
            }
            auto uploaded = chibillm::upload_attention_metadata(
                context, { positions, slots, table, offsets, lengths });
            REQUIRE(uploaded.has_value());
            auto prepared = chibillm::prepared_attention_batch::make(
                context, uploaded->positions, uploaded->block_table, uploaded->table_offsets,
                uploaded->table_lengths, *cache);
            REQUIRE(prepared.has_value());

            chibillm::compute_pass pass(context);
            REQUIRE(pass.begin().has_value());
            auto result = chibillm::run_qwen3_5_full_attention(
                context, config, layer, hidden, uploaded->slots, *prepared, 1, *cache);
            REQUIRE(result.has_value());
            REQUIRE(pass.finish().has_value());
            auto chunk = read_floats(*result);
            actual.insert(actual.end(), chunk.begin(), chunk.end());
        }

        REQUIRE(actual.size() == expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
            CHECK(actual[i] == doctest::Approx(expected[i]).epsilon(1e-4));

        const auto cached_keys = read_floats(cache->keys());
        const auto cached_values = read_floats(cache->values());
        for (std::size_t i = 0; i < cache->elements_per_layer(); ++i) {
            CHECK(cached_keys[i] == 0.0F);
            CHECK(cached_values[i] == 0.0F);
        }
        for (std::size_t row = 0; row < 3; ++row) {
            auto offset = cache->element_offset(1, 1, row, 0, 0);
            REQUIRE(offset.has_value());
            for (std::size_t feature = 0; feature < 4; ++feature) {
                CHECK(cached_keys[*offset + feature]
                      == doctest::Approx(keys[row][feature]).epsilon(1e-4));
                CHECK(cached_values[*offset + feature]
                      == doctest::Approx(values[row][feature]).epsilon(1e-4));
            }
        }
    }
}
