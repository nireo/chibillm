#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "metal_test_support.h"
#include "model_batch.h"
#include "qwen/qwen3_5_model_state.h"
#include "qwen/qwen_layer.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace chibillm;
using namespace metal_test;

namespace {
using vector = std::vector<double>;

qwen3_5_config
small_config()
{
    qwen3_5_config c {};
    c.vocabulary_size = 8;
    c.hidden_size = 4;
    c.intermediate_size = 6;
    c.layer_count = 4;
    c.layer_types = { qwen3_5_layer_type::linear_attention, qwen3_5_layer_type::full_attention,
                      qwen3_5_layer_type::linear_attention, qwen3_5_layer_type::full_attention };
    c.query_head_count = 2;
    c.kv_head_count = 1;
    c.head_dimension = 4;
    c.max_position_embeddings = 32;
    c.rms_epsilon = 0.03F; // Deliberately distinct from the DeltaNet L2 epsilon.
    c.rope_theta = 10000;
    c.partial_rotary_factor = 0.5F;
    c.attention_output_gate = true;
    c.linear_conv_kernel_dimension = 3;
    c.linear_key_head_count = 1;
    c.linear_value_head_count = 2;
    c.linear_key_head_dimension = 3;
    c.linear_value_head_dimension = 2;
    return c;
}

metal_tensor
patterned(const metal_context& context, dtype type, std::vector<std::size_t> shape, float seed)
{
    auto tensor = make_tensor(context, type, std::move(shape));
    std::vector<float> data(tensor.descriptor().element_count());
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = std::sin(float(i) * 1.17F + seed) * 0.5F;
    if (type == dtype::bf16)
        write_bf16(tensor, data);
    else
        write_floats(tensor, data);
    return tensor;
}

qwen3_5_weights
make_weights(const metal_context& context, const qwen3_5_config& c)
{
    float seed = 0;
    auto bf = [&](std::vector<std::size_t> dims) {
        return patterned(context, dtype::bf16, std::move(dims), seed += 0.71F);
    };
    auto fp = [&](std::vector<std::size_t> dims) {
        return patterned(context, dtype::f32, std::move(dims), seed += 0.71F);
    };
    qwen3_5_weights weights { bf({ c.vocabulary_size, c.hidden_size }), bf({ c.hidden_size }), {} };
    for (auto type : c.layer_types) {
        const auto width = 2 * c.linear_key_width() + c.linear_value_width();
        qwen3_5_mixer_weights mixer = type == qwen3_5_layer_type::linear_attention
            ? qwen3_5_mixer_weights(qwen3_5_linear_attention_weights {
                  bf({ width, c.hidden_size }), bf({ c.linear_value_width(), c.hidden_size }),
                  bf({ c.linear_value_head_count, c.hidden_size }),
                  bf({ c.linear_value_head_count, c.hidden_size }),
                  bf({ width, 1, c.linear_conv_kernel_dimension }),
                  fp({ c.linear_value_head_count }), bf({ c.linear_value_head_count }),
                  fp({ c.linear_value_head_dimension }),
                  bf({ c.hidden_size, c.linear_value_width() }) })
            : qwen3_5_mixer_weights(qwen3_5_full_attention_weights {
                  bf({ c.head_dimension }), bf({ c.head_dimension }),
                  bf({ 2 * c.query_width() + 2 * c.kv_width(), c.hidden_size }),
                  bf({ c.hidden_size, c.query_width() }) });
        weights.layers.push_back({ bf({ c.hidden_size }), bf({ c.hidden_size }),
                                   bf({ 2 * c.intermediate_size, c.hidden_size }),
                                   bf({ c.hidden_size, c.intermediate_size }), std::move(mixer) });
    }
    return weights;
}

vector
read_values(const metal_tensor& tensor)
{
    if (tensor.descriptor().type() == dtype::f32) {
        const auto values = read_floats(tensor);
        return { values.begin(), values.end() };
    }
    vector values(tensor.descriptor().element_count());
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint16_t bits;
        std::memcpy(&bits, tensor.buffer().bytes().data() + i * sizeof(bits), sizeof(bits));
        values[i] = bf16::from_bits(bits).to_float();
    }
    return values;
}

vector
project(const vector& input, const metal_tensor& weight)
{
    const auto shape = weight.descriptor().shape().dimensions();
    const auto w = read_values(weight);
    vector output(input.size() / shape[1] * shape[0]);
    for (std::size_t t = 0; t < input.size() / shape[1]; ++t)
        for (std::size_t o = 0; o < shape[0]; ++o)
            for (std::size_t i = 0; i < shape[1]; ++i)
                output[t * shape[0] + o] += input[t * shape[1] + i] * w[o * shape[1] + i];
    return output;
}

vector
normalize(vector input, const metal_tensor& weight, double epsilon, bool zero_centered = true)
{
    const auto w = read_values(weight);
    for (std::size_t base = 0; base < input.size(); base += w.size()) {
        double square = 0;
        for (std::size_t i = 0; i < w.size(); ++i)
            square += input[base + i] * input[base + i];
        for (std::size_t i = 0; i < w.size(); ++i)
            input[base + i] *=
                (w[i] + (zero_centered ? 1 : 0)) / std::sqrt(square / w.size() + epsilon);
    }
    return input;
}

double
silu(double x)
{
    return x / (1 + std::exp(-x));
}

struct reference_layer {
    vector convolution;
    vector recurrent;
    vector keys;
    vector values;
};

struct reference_result {
    vector output;
    std::vector<reference_layer> layers;
};

// Independent, double-precision whole-sequence decoder reference. No GPU ops or
// production batching helpers are used to calculate expected activations/state.
reference_result
reference(const qwen3_5_config& c, const qwen3_5_weights& weights, vector hidden)
{
    const auto rows = hidden.size() / c.hidden_size;
    reference_result result { {}, {} };
    for (const auto& layer : weights.layers) {
        auto input = normalize(hidden, layer.input_norm, c.rms_epsilon);
        reference_layer memory;
        vector mixed;
        const metal_tensor* output_weight;
        if (const auto* a = std::get_if<qwen3_5_linear_attention_weights>(&layer.mixer)) {
            const auto kw = c.linear_key_width(), vw = c.linear_value_width(), width = 2 * kw + vw;
            const auto kh = c.linear_key_head_count, vh = c.linear_value_head_count;
            const auto kd = c.linear_key_head_dimension, vd = c.linear_value_head_dimension;
            const auto kernel = c.linear_conv_kernel_dimension;
            const auto qkv = project(input, a->qkv_projection);
            const auto gate = project(input, a->gate_projection);
            const auto decay = project(input, a->decay_projection);
            const auto beta = project(input, a->learning_rate_projection);
            const auto conv_weight = read_values(a->convolution);
            const auto logs = read_values(a->decay_log), bias = read_values(a->learning_rate_bias);
            vector convolved(qkv.size());
            memory.convolution.resize(width * kernel);
            for (std::size_t t = 0; t < rows; ++t)
                for (std::size_t f = 0; f < width; ++f) {
                    double sum = 0;
                    for (std::size_t k = 0; k < kernel; ++k)
                        if (t + k + 1 >= kernel)
                            sum +=
                                qkv[(t + k + 1 - kernel) * width + f] * conv_weight[f * kernel + k];
                    convolved[t * width + f] = silu(sum);
                }
            for (std::size_t f = 0; f < width; ++f)
                for (std::size_t k = 0; k < kernel; ++k)
                    if (rows + k >= kernel)
                        memory.convolution[f * kernel + k] = qkv[(rows + k - kernel) * width + f];
            memory.recurrent.resize(vh * kd * vd);
            mixed.resize(rows * vw);
            for (std::size_t t = 0; t < rows; ++t)
                for (std::size_t h = 0; h < vh; ++h) {
                    vector q(kd), k(kd), correction(vd);
                    double qsum = 1e-6, ksum = 1e-6;
                    for (std::size_t i = 0; i < kd; ++i) {
                        const auto pos = t * width + (h / (vh / kh)) * kd + i;
                        q[i] = convolved[pos];
                        k[i] = convolved[pos + kw];
                        qsum += q[i] * q[i];
                        ksum += k[i] * k[i];
                    }
                    for (std::size_t i = 0; i < kd; ++i) {
                        q[i] /= std::sqrt(qsum * kd);
                        k[i] /= std::sqrt(ksum);
                    }
                    const double g = std::exp(-std::exp(logs[h])
                                              * std::log1p(std::exp(decay[t * vh + h] + bias[h])));
                    const double b = 1 / (1 + std::exp(-beta[t * vh + h]));
                    const auto base = h * kd * vd;
                    for (std::size_t i = 0; i < kd * vd; ++i)
                        memory.recurrent[base + i] *= g;
                    for (std::size_t v = 0; v < vd; ++v) {
                        double predicted = 0;
                        for (std::size_t i = 0; i < kd; ++i)
                            predicted += k[i] * memory.recurrent[base + i * vd + v];
                        correction[v] =
                            b * (convolved[t * width + 2 * kw + h * vd + v] - predicted);
                    }
                    for (std::size_t i = 0; i < kd; ++i)
                        for (std::size_t v = 0; v < vd; ++v) {
                            memory.recurrent[base + i * vd + v] += k[i] * correction[v];
                            mixed[t * vw + h * vd + v] +=
                                q[i] * memory.recurrent[base + i * vd + v];
                        }
                }
            mixed = normalize(std::move(mixed), a->norm, c.rms_epsilon, false);
            for (std::size_t i = 0; i < mixed.size(); ++i)
                mixed[i] *= silu(gate[i]);
            output_weight = &a->output;
        } else {
            const auto& full = std::get<qwen3_5_full_attention_weights>(layer.mixer);
            const auto projected = project(input, full.qkv_packed);
            const auto qw = c.query_width(), kw = c.kv_width(), d = c.head_dimension;
            const auto packed = 2 * qw + 2 * kw;
            vector queries(rows * qw), gates(rows * qw);
            memory.keys.resize(rows * kw);
            memory.values.resize(rows * kw);
            for (std::size_t t = 0; t < rows; ++t) {
                for (std::size_t h = 0; h < c.query_head_count; ++h)
                    for (std::size_t i = 0; i < d; ++i) {
                        queries[t * qw + h * d + i] = projected[t * packed + h * 2 * d + i];
                        gates[t * qw + h * d + i] = projected[t * packed + h * 2 * d + d + i];
                    }
                for (std::size_t i = 0; i < kw; ++i) {
                    memory.keys[t * kw + i] = projected[t * packed + 2 * qw + i];
                    memory.values[t * kw + i] = projected[t * packed + 2 * qw + kw + i];
                }
            }
            queries = normalize(std::move(queries), full.query_norm, c.rms_epsilon);
            memory.keys = normalize(std::move(memory.keys), full.key_norm, c.rms_epsilon);
            auto rotate = [&](vector& x, std::size_t heads) {
                for (std::size_t t = 0; t < rows; ++t)
                    for (std::size_t h = 0; h < heads; ++h)
                        for (std::size_t i = 0; i < c.rotary_dimension() / 2; ++i) {
                            const auto base = (t * heads + h) * d;
                            const auto other = i + c.rotary_dimension() / 2;
                            const auto angle =
                                t / std::pow(c.rope_theta, 2.0 * i / c.rotary_dimension());
                            const auto first = x[base + i], second = x[base + other];
                            x[base + i] = first * std::cos(angle) - second * std::sin(angle);
                            x[base + other] = second * std::cos(angle) + first * std::sin(angle);
                        }
            };
            rotate(queries, c.query_head_count);
            rotate(memory.keys, c.kv_head_count);
            mixed.resize(rows * qw);
            for (std::size_t t = 0; t < rows; ++t)
                for (std::size_t h = 0; h < c.query_head_count; ++h) {
                    const auto kvh = h / (c.query_head_count / c.kv_head_count);
                    vector scores(t + 1);
                    for (std::size_t s = 0; s <= t; ++s)
                        for (std::size_t i = 0; i < d; ++i)
                            scores[s] += queries[t * qw + h * d + i]
                                * memory.keys[s * kw + kvh * d + i]
                                / std::sqrt(double(d));
                    const auto maximum = *std::max_element(scores.begin(), scores.end());
                    double total = 0;
                    for (auto& score : scores)
                        total += score = std::exp(score - maximum);
                    for (std::size_t i = 0; i < d; ++i) {
                        double value = 0;
                        for (std::size_t s = 0; s <= t; ++s)
                            value += scores[s] / total * memory.values[s * kw + kvh * d + i];
                        mixed[t * qw + h * d + i] =
                            value / (1 + std::exp(-gates[t * qw + h * d + i]));
                    }
                }
            output_weight = &full.output;
        }
        auto residual = project(mixed, *output_weight);
        for (std::size_t i = 0; i < hidden.size(); ++i)
            hidden[i] += residual[i];
        auto gateup = project(normalize(hidden, layer.post_attention_norm, c.rms_epsilon),
                              layer.gateup_packed);
        vector activated(rows * c.intermediate_size);
        for (std::size_t t = 0; t < rows; ++t)
            for (std::size_t i = 0; i < c.intermediate_size; ++i)
                activated[t * c.intermediate_size + i] =
                    silu(gateup[t * 2 * c.intermediate_size + i])
                    * gateup[t * 2 * c.intermediate_size + c.intermediate_size + i];
        residual = project(activated, layer.mlp_down);
        for (std::size_t i = 0; i < hidden.size(); ++i)
            hidden[i] += residual[i];
        result.layers.push_back(std::move(memory));
    }
    result.output = std::move(hidden);
    return result;
}

vector
input_for(seq_id id, std::size_t rows)
{
    vector input(rows * 4);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = float(std::sin(double(i) * 0.37 + double(id)) * 1.5);
    return input;
}

void
near(const vector& actual, const vector& expected)
{
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CAPTURE(i);
        CHECK(std::isfinite(actual[i]));
        CHECK(std::abs(actual[i] - expected[i]) <= 2e-5 + 3e-4 * std::abs(expected[i]));
    }
}

std::unique_ptr<qwen3_5_model_state>
make_state(const metal_context& context, const qwen3_5_config& config)
{
    auto state = qwen3_5_model_state::make(context, config, 12, 2);
    REQUIRE(state);
    // Reserve in reverse order, so sequence identity is neither batch position
    // nor physical page number.
    REQUIRE((*state)->reserve(29, 6));
    REQUIRE((*state)->reserve(11, 6));
    return std::move(*state);
}

model_batch
batch_for(const qwen3_5_model_state& state,
          std::initializer_list<std::pair<seq_id, std::size_t>> chunks,
          batch_phase phase = batch_phase::prefill)
{
    model_batch batch { .id = 1,
                        .phase = phase,
                        .kv_block_size = state.block_size(),
                        .tokens = {},
                        .positions = {},
                        .items = {} };
    for (auto [id, count] : chunks) {
        const auto offset = batch.token_count();
        const auto start = *state.committed_tokens(id);
        const auto blocks = state.resources(id).blocks;
        batch.items.push_back(
            { id, offset, count, std::nullopt, { blocks.begin(), blocks.end() } });
        for (std::size_t i = 0; i < count; ++i) {
            batch.tokens.push_back(1);
            batch.positions.push_back(static_cast<position_id>(start + i));
        }
    }
    return batch;
}

metal_tensor
hidden_for(const metal_context& context, const model_batch& batch)
{
    std::vector<float> data;
    for (const auto& item : batch.items) {
        const auto start = batch.positions[item.token_offset];
        const auto input = input_for(item.id, start + item.token_count);
        data.insert(data.end(), input.begin() + start * 4, input.end());
    }
    auto tensor = make_tensor(context, dtype::f32, { batch.token_count(), 4 });
    write_floats(tensor, data);
    return tensor;
}

void
execute_and_check(metal_context& context,
                  const qwen3_5_config& config,
                  const qwen3_5_weights& weights,
                  qwen3_5_model_state& state,
                  const model_batch& batch)
{
    vector expected;
    for (const auto& item : batch.items) {
        const auto start = batch.positions[item.token_offset];
        auto ref = reference(config, weights, input_for(item.id, start + item.token_count));
        expected.insert(expected.end(), ref.output.begin() + start * config.hidden_size,
                        ref.output.end());
    }
    REQUIRE(state.begin_batch(batch));
    compute_pass pass(context);
    REQUIRE(pass.begin());
    auto output =
        run_qwen3_5_layers(context, config, weights, hidden_for(context, batch), batch, state);
    REQUIRE(output);
    REQUIRE(pass.finish());
    // The layer loop must leave transaction control to its caller.
    for (const auto& item : batch.items)
        CHECK(state.committed_tokens(item.id) == batch.positions[item.token_offset]);
    state.commit_batch();
    near(read_values(*output), expected);
}

void
check_state(const qwen3_5_config& c,
            const qwen3_5_weights& weights,
            const qwen3_5_model_state& state,
            seq_id id,
            std::size_t rows)
{
    const auto ref = reference(c, weights, input_for(id, rows));
    CHECK(state.committed_tokens(id) == rows);
    for (std::size_t layer = 0; layer < c.layer_count; ++layer) {
        CAPTURE(id);
        CAPTURE(layer);
        const auto& expected = ref.layers[layer];
        if (const auto* memory = state.linear_state(id, layer)) {
            near(read_values(memory->convolution), expected.convolution);
            near(read_values(memory->recurrent), expected.recurrent);
        } else {
            const auto keys = read_values(state.cache().keys()),
                       values = read_values(state.cache().values());
            for (std::size_t t = 0; t < rows; ++t) {
                const auto block = state.resources(id).blocks[t / state.block_size()];
                const auto offset = state.cache().element_offset(*state.cache_layer(layer), block,
                                                                 t % state.block_size(), 0, 0);
                REQUIRE(offset);
                near(vector(keys.begin() + *offset, keys.begin() + *offset + c.kv_width()),
                     vector(expected.keys.begin() + t * c.kv_width(),
                            expected.keys.begin() + (t + 1) * c.kv_width()));
                near(vector(values.begin() + *offset, values.begin() + *offset + c.kv_width()),
                     vector(expected.values.begin() + t * c.kv_width(),
                            expected.values.begin() + (t + 1) * c.kv_width()));
            }
        }
    }
}

std::vector<std::byte>
snapshot_linear(const qwen3_5_model_state& state, const qwen3_5_config& config)
{
    std::vector<std::byte> bytes;
    for (seq_id id : { 11, 29 })
        for (std::size_t layer = 0; layer < config.layer_count; ++layer)
            if (const auto* memory = state.linear_state(id, layer)) {
                for (const auto* tensor : { &memory->convolution, &memory->recurrent }) {
                    const auto data = tensor->buffer().bytes();
                    bytes.insert(bytes.end(), data.begin(), data.end());
                }
            }
    return bytes;
}
} // namespace

TEST_CASE("Qwen3.5 hybrid layers match a CPU decoder with grouped heads and compact KV layers")
{
    auto made = metal_context::make(load_shader_source());
    REQUIRE(made);
    auto& context = *made;
    const auto config = small_config();
    const auto weights = make_weights(context, config);
    for (const bool chunked : { false, true }) {
        CAPTURE(chunked);
        auto state = make_state(context, config);
        if (chunked) {
            execute_and_check(context, config, weights, *state,
                              batch_for(*state, { { 11, 2 }, { 29, 1 } }));
            execute_and_check(context, config, weights, *state,
                              batch_for(*state, { { 29, 2 }, { 11, 1 } }));
            execute_and_check(context, config, weights, *state,
                              batch_for(*state, { { 11, 1 }, { 29, 1 } }, batch_phase::decode));
            execute_and_check(context, config, weights, *state,
                              batch_for(*state, { { 11, 1 } }, batch_phase::decode));
        } else {
            execute_and_check(context, config, weights, *state,
                              batch_for(*state, { { 11, 5 }, { 29, 4 } }));
        }
        check_state(config, weights, *state, 11, 5);
        check_state(config, weights, *state, 29, 4);
    }
}

TEST_CASE("Qwen3.5 hybrid execution can be aborted and retried after partial GPU work")
{
    auto made = metal_context::make(load_shader_source());
    REQUIRE(made);
    auto& context = *made;
    const auto config = small_config();
    auto weights = make_weights(context, config);
    for (const bool late_failure : { false, true }) {
        CAPTURE(late_failure);
        auto state = make_state(context, config);
        execute_and_check(context, config, weights, *state,
                          batch_for(*state, { { 11, 1 }, { 29, 1 } }));
        const auto saved = snapshot_linear(*state, config);
        const auto batch = batch_for(*state, { { 29, 2 }, { 11, 1 } });
        REQUIRE(state->begin_batch(batch));
        std::optional<metal_tensor> down;
        if (late_failure) {
            // Fail in the very last MLP, after both recurrent and KV layers have
            // encoded writes. The layer executor must not commit early.
            down = std::move(weights.layers.back().mlp_down);
            weights.layers.back().mlp_down = make_tensor(
                context, dtype::bf16, { config.hidden_size, config.intermediate_size + 1 });
        }
        {
            compute_pass pass(context);
            REQUIRE(pass.begin());
            auto result = run_qwen3_5_layers(context, config, weights, hidden_for(context, batch),
                                             batch, *state);
            CHECK(result.has_value() == !late_failure);
            if (!late_failure)
                REQUIRE(pass.finish());
            // On failure, the pass destructor must drain encoded GPU writes
            // before abort_batch restores the CPU-visible state.
        }
        if (down)
            weights.layers.back().mlp_down = std::move(*down);
        CHECK(snapshot_linear(*state, config) != saved);
        CHECK(state->committed_tokens(11) == 1);
        CHECK(state->committed_tokens(29) == 1);
        state->abort_batch();
        CHECK(snapshot_linear(*state, config) == saved);
        execute_and_check(context, config, weights, *state, batch);
        check_state(config, weights, *state, 11, 2);
        check_state(config, weights, *state, 29, 3);
    }
}

TEST_CASE("Qwen3.5 hybrid layers reject invalid routing before mutating state")
{
    auto made = metal_context::make(load_shader_source());
    REQUIRE(made);
    auto& context = *made;
    const auto config = small_config();
    auto weights = make_weights(context, config);
    auto state = make_state(context, config);
    execute_and_check(context, config, weights, *state,
                      batch_for(*state, { { 11, 1 }, { 29, 1 } }));
    const auto saved = snapshot_linear(*state, config);
    auto batch = batch_for(*state, { { 11, 2 }, { 29, 1 } });
    REQUIRE(state->begin_batch(batch));
    auto hidden = hidden_for(context, batch);
    auto invalid_config = config;

    SUBCASE("duplicate sequence")
    {
        batch.items[1].id = batch.items[0].id;
        batch.items[1].block_table = batch.items[0].block_table;
    }
    SUBCASE("unknown sequence")
    {
        batch.items[1].id = 99;
    }
    SUBCASE("foreign blocks")
    {
        std::swap(batch.items[0].block_table, batch.items[1].block_table);
    }
    SUBCASE("overflowing range")
    {
        batch.items[1].token_count = std::numeric_limits<std::size_t>::max();
    }
    SUBCASE("wrong row offset")
    {
        ++batch.items[1].token_offset;
    }
    SUBCASE("uncovered rows")
    {
        batch.items.pop_back();
    }
    SUBCASE("replayed prefix")
    {
        batch.positions[0] = 0;
    }
    SUBCASE("nonconsecutive chunk")
    {
        ++batch.positions[1];
    }
    SUBCASE("multi-token decode")
    {
        batch.phase = batch_phase::decode;
    }
    SUBCASE("wrong hidden shape")
    {
        hidden = make_tensor(context, dtype::f32, { batch.token_count(), 5 });
    }
    SUBCASE("wrong hidden dtype")
    {
        hidden = make_tensor(context, dtype::bf16, { batch.token_count(), 4 });
    }
    SUBCASE("wrong mixer variant")
    {
        std::swap(weights.layers[0].mixer, weights.layers[1].mixer);
    }
    SUBCASE("wrong cache layer mapping")
    {
        std::swap(invalid_config.layer_types[0], invalid_config.layer_types[1]);
        std::swap(weights.layers[0].mixer, weights.layers[1].mixer);
    }
    SUBCASE("missing weights")
    {
        weights.layers.pop_back();
    }
    SUBCASE("wrong recurrent geometry")
    {
        ++invalid_config.linear_value_head_dimension;
    }
    SUBCASE("missing layer types")
    {
        invalid_config.layer_types.pop_back();
    }

    compute_pass pass(context);
    REQUIRE(pass.begin());
    auto result =
        run_qwen3_5_layers(context, invalid_config, weights, std::move(hidden), batch, *state);
    REQUIRE_FALSE(result);
    CHECK(result.error() == tensor_op_errc::input_shape_mismatch);
    REQUIRE(pass.finish());
    CHECK(snapshot_linear(*state, config) == saved);
    state->abort_batch();
}
