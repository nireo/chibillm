#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "metal/metal_kernels.h"
#include "metal_test_support.h"
#include "tensor/deltanet.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

using namespace chibillm;
using namespace metal_test;

namespace {
std::vector<float>
values(std::size_t count, float phase = 0)
{
    std::vector<float> result(count);
    for (std::size_t i = 0; i < count; ++i)
        result[i] = std::sin(float(i) * 0.71F + phase) * 0.75F;
    return result;
}

metal_tensor
f32(const metal_context& context, std::vector<std::size_t> shape, const std::vector<float>& data)
{
    auto tensor = make_tensor(context, dtype::f32, std::move(shape));
    write_floats(tensor, data);
    return tensor;
}

std::vector<float>
round_bf16(std::vector<float> data)
{
    for (auto& v : data)
        v = bf16::from_float(v).to_float();
    return data;
}

void
near(const metal_tensor& tensor, const std::vector<float>& expected)
{
    const auto actual = read_floats(tensor);
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CAPTURE(i);
        CHECK(std::isfinite(actual[i]));
        CHECK(std::abs(actual[i] - expected[i]) <= 2e-6F + 2e-4F * std::abs(expected[i]));
    }
}

std::vector<float>
slice(const std::vector<float>& data, std::size_t begin, std::size_t end)
{
    return { data.begin() + begin, data.begin() + end };
}

// CPU reference: concatenate history and inputs, then perform a causal cross-
// correlation. Uses double accumulation and does not mirror the GPU's shift loop.
std::vector<float>
conv_reference(const std::vector<float>& input,
               const std::vector<float>& weight,
               std::vector<float>& history,
               std::size_t rows,
               std::size_t channels,
               std::size_t kernel)
{
    std::vector<float> output(input.size());
    for (std::size_t c = 0; c < channels; ++c) {
        auto sequence = slice(history, c * kernel, (c + 1) * kernel);
        for (std::size_t t = 0; t < rows; ++t)
            sequence.push_back(input[t * channels + c]);
        for (std::size_t t = 0; t < rows; ++t) {
            double sum = 0;
            for (std::size_t j = 0; j < kernel; ++j)
                sum += double(sequence[t + 1 + j]) * weight[c * kernel + j];
            output[t * channels + c] = float(sum / (1 + std::exp(-sum)));
        }
        std::copy(sequence.end() - kernel, sequence.end(), history.begin() + c * kernel);
    }
    return output;
}

// Double-precision matrix reference: form the entire decayed state and rank-one
// update before multiplying by Q. Checks update order, scaling, and head mapping.
std::vector<float>
delta_reference(const std::vector<float>& qkv,
                const std::vector<float>& a,
                const std::vector<float>& b,
                const std::vector<float>& logs,
                const std::vector<float>& bias,
                std::vector<float>& state,
                std::size_t rows,
                std::size_t kh,
                std::size_t vh,
                std::size_t kd,
                std::size_t vd)
{
    std::vector<double> matrix(state.begin(), state.end());
    const auto kw = kh * kd, vw = vh * vd, packed = 2 * kw + vw;
    std::vector<float> output(rows * vw);
    for (std::size_t t = 0; t < rows; ++t) {
        for (std::size_t h = 0; h < vh; ++h) {
            std::vector<double> q(kd), k(kd), correction(vd);
            double qnorm = 1e-6, knorm = 1e-6;
            for (std::size_t j = 0; j < kd; ++j) {
                const auto index = t * packed + (h / (vh / kh)) * kd + j;
                q[j] = qkv[index];
                k[j] = qkv[index + kw];
                qnorm += q[j] * q[j];
                knorm += k[j] * k[j];
            }
            for (std::size_t j = 0; j < kd; ++j) {
                q[j] /= std::sqrt(qnorm) * std::sqrt(double(kd));
                k[j] /= std::sqrt(knorm);
            }
            const double x = double(a[t * vh + h]) + bias[h];
            const double decay =
                std::exp(-std::exp(double(logs[h]))
                         * (std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)))));
            const double beta = 1 / (1 + std::exp(-double(b[t * vh + h])));
            const auto base = h * kd * vd;
            for (std::size_t j = 0; j < kd * vd; ++j)
                matrix[base + j] *= decay;
            for (std::size_t v = 0; v < vd; ++v) {
                double prediction = 0;
                for (std::size_t j = 0; j < kd; ++j)
                    prediction += k[j] * matrix[base + j * vd + v];
                correction[v] = beta * (qkv[t * packed + 2 * kw + h * vd + v] - prediction);
            }
            for (std::size_t j = 0; j < kd; ++j)
                for (std::size_t v = 0; v < vd; ++v)
                    matrix[base + j * vd + v] += k[j] * correction[v];
            for (std::size_t v = 0; v < vd; ++v) {
                double result = 0;
                for (std::size_t j = 0; j < kd; ++j)
                    result += q[j] * matrix[base + j * vd + v];
                output[t * vw + h * vd + v] = float(result);
            }
        }
    }
    std::copy(matrix.begin(), matrix.end(), state.begin());
    return output;
}
} // namespace

TEST_CASE("DeltaNet convolution matches CPU with raw history across arbitrary chunks")
{
    const auto& context = test_context();
    for (const std::size_t kernel : { 1, 4 }) {
        constexpr std::size_t rows = 7, channels = 5;
        const auto input_data = values(rows * channels);
        const auto weights = round_bf16(values(channels * kernel, 1));
        const auto initial = values(channels * kernel, 2);
        auto expected_history = initial;
        const auto expected =
            conv_reference(input_data, weights, expected_history, rows, channels, kernel);
        auto input = f32(context, { rows, channels }, input_data);
        auto weight = make_tensor(context, dtype::bf16, { channels, 1, kernel });
        write_bf16(weight, weights);
        auto history = f32(context, { channels, kernel }, initial);
        auto output = make_tensor(context, dtype::f32, { rows, channels });
        REQUIRE(causal_conv1d_silu(context, input, weight, history, output));
        near(output, expected);
        near(history, expected_history);

        write_floats(history, initial);
        std::size_t offset = 0;
        for (const std::size_t count : { 2, 1, 4 }) {
            auto chunk = f32(context, { count, channels },
                             slice(input_data, offset * channels, (offset + count) * channels));
            auto result = make_tensor(context, dtype::f32, { count, channels });
            REQUIRE(causal_conv1d_silu(context, chunk, weight, history, result));
            near(result, slice(expected, offset * channels, (offset + count) * channels));
            offset += count;
        }
        near(history, expected_history);
    }
}

TEST_CASE(
    "DeltaNet recurrence matches CPU and chunked decode for grouped heads and real head dimensions")
{
    const auto& context = test_context();
    for (const std::size_t kd : { 3, 128 }) {
        const std::size_t rows = 5, kh = 2, vh = 4, vd = kd == 3 ? 5 : 128;
        const auto width = 2 * kh * kd + vh * vd;
        auto packed = values(rows * width);
        // A zero Q/K head tests epsilon handling without NaNs.
        std::fill(packed.begin(), packed.begin() + kd, 0);
        std::fill(packed.begin() + kh * kd, packed.begin() + kh * kd + kd, 0);
        auto a_data = values(rows * vh, 1), b_data = values(rows * vh, 2);
        a_data[0] = 100;
        a_data[1] = -100;
        b_data[2] = 100;
        b_data[3] = -100;
        const auto logs_data = values(vh, 3), bias_data = round_bf16(values(vh, 4));
        const auto initial = values(vh * kd * vd, 5);
        auto expected_state = initial;
        const auto expected = delta_reference(packed, a_data, b_data, logs_data, bias_data,
                                              expected_state, rows, kh, vh, kd, vd);
        auto qkv = f32(context, { rows, width }, packed);
        auto a = f32(context, { rows, vh }, a_data);
        auto b = f32(context, { rows, vh }, b_data);
        auto logs = f32(context, { vh }, logs_data);
        auto bias = make_tensor(context, dtype::bf16, { vh });
        write_bf16(bias, bias_data);
        auto state = f32(context, { vh, kd, vd }, initial);
        auto output = make_tensor(context, dtype::f32, { rows, vh * vd });
        REQUIRE(gated_delta_rule(context, qkv, a, b, logs, bias, kh, state, output));
        near(output, expected);
        near(state, expected_state);
        write_floats(state, initial);
        std::size_t offset = 0;
        for (const std::size_t count : { 2, 1, 2 }) {
            auto chunk = f32(context, { count, width },
                             slice(packed, offset * width, (offset + count) * width));
            auto ac =
                f32(context, { count, vh }, slice(a_data, offset * vh, (offset + count) * vh));
            auto bc =
                f32(context, { count, vh }, slice(b_data, offset * vh, (offset + count) * vh));
            auto result = make_tensor(context, dtype::f32, { count, vh * vd });
            REQUIRE(gated_delta_rule(context, chunk, ac, bc, logs, bias, kh, state, result));
            near(result, slice(expected, offset * vh * vd, (offset + count) * vh * vd));
            offset += count;
        }
        near(state, expected_state);
    }
}

TEST_CASE("DeltaNet chunkwise prefill matches CPU and sequential GPU including continuation")
{
    auto made = metal_context::make(load_shader_source());
    REQUIRE(made);
    auto& context = *made;
    for (const std::size_t kd : { 3, 128 }) {
        for (const std::size_t count : { 31, 32, 33, 65, 129, 512 }) {
            CAPTURE(kd);
            CAPTURE(count);
            constexpr std::size_t kh = 2, vh = 4, offset = 2;
            const std::size_t vd = kd == 3 ? 5 : 128;
            const auto rows = count + offset + 1, width = 2 * kh * kd + vh * vd;
            auto packed = values(rows * width);
            if (count == 512) {
                // Deterministic broad-spectrum inputs in addition to the
                // strongly correlated sinusoidal cases above.
                std::uint32_t seed = 42;
                for (auto& value : packed) {
                    seed = seed * 1664525U + 1013904223U;
                    value = float(seed >> 8) / 8388608.0F - 1.0F;
                }
            }
            // Exercise zeros, nearly-collinear keys, and extreme decay/beta.
            std::fill(packed.begin() + offset * width,
                      packed.begin() + offset * width + 2 * kh * kd, 0);
            auto av = values(rows * vh, 1), bv = values(rows * vh, 2);
            av[(offset + 17) * vh] = 1000;
            av[(offset + 18) * vh + 1] = -100;
            bv[(offset + 19) * vh] = 100;
            bv[(offset + 19) * vh + 1] = -100;
            const std::vector<float> lv { -6, -2, 0, 2 };
            const auto biasv = round_bf16(values(vh, 4));
            const auto initial =
                count == 33 ? std::vector<float>(vh * kd * vd, 0) : values(vh * kd * vd, 5);
            auto cpu_state = initial;
            const auto expected =
                delta_reference(slice(packed, offset * width, (offset + count + 1) * width),
                                slice(av, offset * vh, (offset + count + 1) * vh),
                                slice(bv, offset * vh, (offset + count + 1) * vh), lv, biasv,
                                cpu_state, count + 1, kh, vh, kd, vd);
            auto qkv = f32(context, { rows, width }, packed);
            auto a = f32(context, { rows, vh }, av), b = f32(context, { rows, vh }, bv);
            auto logs = f32(context, { vh }, lv);
            auto bias = make_tensor(context, dtype::bf16, { vh });
            write_bf16(bias, biasv);
            auto state = f32(context, { vh, kd, vd }, initial);
            auto reference_state = f32(context, { vh, kd, vd }, initial);
            std::vector<float> sentinels(rows * vh * vd, -123);
            auto output = f32(context, { rows, vh * vd }, sentinels);
            auto reference_output = f32(context, { rows, vh * vd }, sentinels);
            compute_pass pass(context);
            REQUIRE(pass.begin());
            REQUIRE(gated_delta_rule(context, qkv, a, b, logs, bias, kh, state, output, 1e-6F,
                                     deltanet_chunk { offset, count }));
            REQUIRE(metal_kernels(context).dispatch_gated_delta_rule(
                qkv.buffer(), a.buffer(), b.buffer(), logs.buffer(), bias.buffer(),
                reference_state.buffer(), reference_output.buffer(), count, kh, vh, kd, vd, 1e-6F,
                offset, false));
            REQUIRE(pass.finish());
            near(state, read_floats(reference_state));
            near(output, read_floats(reference_output));
            // The untouched suffix becomes a single-token decode using the
            // state returned by prefill, in a later command buffer.
            REQUIRE(gated_delta_rule(context, qkv, a, b, logs, bias, kh, state, output, 1e-6F,
                                     deltanet_chunk { offset + count, 1 }));
            std::copy(expected.begin(), expected.end(), sentinels.begin() + offset * vh * vd);
            near(output, sentinels);
            near(state, cpu_state);
        }
    }
}

TEST_CASE("DeltaNet chunkwise prefill composes across independent states and scratch reuse")
{
    auto made = metal_context::make(load_shader_source());
    REQUIRE(made);
    auto& context = *made;
    constexpr std::size_t rows = 160, kh = 1, vh = 2, kd = 7, vd = 9;
    constexpr std::size_t width = 2 * kh * kd + vh * vd;
    auto qkv = f32(context, { rows, width }, values(rows * width));
    auto a = f32(context, { rows, vh }, values(rows * vh, 1));
    auto b = f32(context, { rows, vh }, values(rows * vh, 2));
    auto logs = f32(context, { vh }, { -4, -1 });
    auto bias = make_tensor(context, dtype::bf16, { vh });
    write_bf16(bias, { 0, 0 });
    auto initial = values(vh * kd * vd, 3);
    auto state = f32(context, { vh, kd, vd }, initial);
    auto other = f32(context, { vh, kd, vd }, initial);
    auto reference = f32(context, { vh, kd, vd }, initial);
    auto out = f32(context, { rows, vh * vd }, std::vector<float>(rows * vh * vd, -123));
    auto other_out = f32(context, { rows, vh * vd }, read_floats(out));
    auto expected = f32(context, { rows, vh * vd }, read_floats(out));
    REQUIRE(metal_kernels(context).dispatch_gated_delta_rule(
        qkv.buffer(), a.buffer(), b.buffer(), logs.buffer(), bias.buffer(), reference.buffer(),
        expected.buffer(), 130, kh, vh, kd, vd, 1e-6F, 1, false));
    compute_pass pass(context);
    REQUIRE(pass.begin());
    REQUIRE(gated_delta_rule(context, qkv, a, b, logs, bias, kh, state, out, 1e-6F,
                             deltanet_chunk { 1, 65 }));
    REQUIRE(gated_delta_rule(context, qkv, a, b, logs, bias, kh, other, other_out, 1e-6F,
                             deltanet_chunk { 1, 130 }));
    REQUIRE(gated_delta_rule(context, qkv, a, b, logs, bias, kh, state, out, 1e-6F,
                             deltanet_chunk { 66, 65 }));
    REQUIRE(pass.finish());
    near(state, read_floats(reference));
    near(other, read_floats(reference));
    near(out, read_floats(expected));
    near(other_out, read_floats(expected));
}

TEST_CASE("DeltaNet kernels compose within one compute pass with separate sequence states")
{
    auto made = metal_context::make(load_shader_source());
    REQUIRE(made);
    auto& context = *made;
    constexpr std::size_t rows = 3, width = 8;
    const auto inputs = values(rows * width);
    const auto weights = round_bf16(values(width * 4, 1));
    std::vector<float> history_data(width * 4, 0), state_data(2 * 2, 0);
    const auto convolved = conv_reference(inputs, weights, history_data, rows, width, 4);
    const std::vector<float> a_data(rows, 0.1F), b_data(rows, -0.2F), logs_data { -1 },
        bias_data { 0.5F };
    // key_dim=2, value_dim=4, one head => 8 packed channels.
    state_data.resize(8, 0);
    auto expected = delta_reference(convolved, a_data, b_data, logs_data, bias_data, state_data,
                                    rows, 1, 1, 2, 4);
    const auto gates = values(rows * 4, 3);
    const std::vector<float> norm_weights { 0, 0.5F, 1, -1 };
    for (std::size_t t = 0; t < rows; ++t) {
        double square = 0;
        for (std::size_t j = 0; j < 4; ++j)
            square += double(expected[t * 4 + j]) * expected[t * 4 + j];
        for (std::size_t j = 0; j < 4; ++j) {
            const auto i = t * 4 + j;
            expected[i] = float(expected[i]
                                / std::sqrt(square / 4 + 1e-6)
                                * norm_weights[j]
                                * gates[i]
                                / (1 + std::exp(-double(gates[i]))));
        }
    }
    auto input = f32(context, { rows, width }, inputs);
    auto weight = make_tensor(context, dtype::bf16, { width, 1, 4 });
    write_bf16(weight, weights);
    auto a = f32(context, { rows, 1 }, a_data), b = f32(context, { rows, 1 }, b_data);
    auto logs = f32(context, { 1 }, logs_data);
    auto bias = make_tensor(context, dtype::bf16, { 1 });
    write_bf16(bias, bias_data);
    auto gate = f32(context, { rows, 4 }, gates), norm = f32(context, { 4 }, norm_weights);
    // Reusing immutable weights for two sequences must not carry state between them.
    for (int sequence = 0; sequence < 2; ++sequence) {
        auto history = f32(context, { width, 4 }, std::vector<float>(width * 4, 0));
        auto state = f32(context, { 1, 2, 4 }, std::vector<float>(8, 0));
        auto conv = make_tensor(context, dtype::f32, { rows, width });
        auto delta = make_tensor(context, dtype::f32, { rows, 4 });
        auto output = make_tensor(context, dtype::f32, { rows, 4 });
        compute_pass pass(context);
        REQUIRE(pass.begin());
        REQUIRE(causal_conv1d_silu(context, input, weight, history, conv));
        REQUIRE(gated_delta_rule(context, conv, a, b, logs, bias, 1, state, delta));
        REQUIRE(rms_norm_gated(context, delta, gate, norm, 1e-6F, output));
        REQUIRE(pass.finish());
        near(output, expected);
        near(state, state_data);
        near(history, history_data);
    }
}

TEST_CASE("DeltaNet batch chunks match standalone sequences in one compute pass")
{
    auto made = metal_context::make(load_shader_source());
    REQUIRE(made);
    auto& context = *made;
    constexpr std::size_t rows = 8, kh = 2, vh = 4, kd = 3, vd = 5, kernel = 4;
    constexpr std::size_t width = 2 * kh * kd + vh * vd, value_width = vh * vd;
    constexpr float sentinel = -123.0F;
    const std::array chunks { deltanet_chunk { 1, 2 }, deltanet_chunk { 4, 3 } };
    const auto input_data = values(rows * width);
    const auto a_data = values(rows * vh, 1), b_data = values(rows * vh, 2);
    auto input = f32(context, { rows, width }, input_data);
    auto weight = make_tensor(context, dtype::bf16, { width, 1, kernel });
    write_bf16(weight, values(width * kernel, 3));
    auto a = f32(context, { rows, vh }, a_data), b = f32(context, { rows, vh }, b_data);
    auto logs = f32(context, { vh }, values(vh, 4));
    auto bias = make_tensor(context, dtype::bf16, { vh });
    write_bf16(bias, values(vh, 5));
    std::vector<float> expected_conv(rows * width, sentinel);
    std::vector<float> expected_delta(rows * value_width, sentinel);
    auto conv = f32(context, { rows, width }, expected_conv);
    auto delta = f32(context, { rows, value_width }, expected_delta);
    std::vector<metal_tensor> histories, states;
    std::vector<std::vector<float>> expected_histories, expected_states;

    for (std::size_t i = 0; i < chunks.size(); ++i) {
        const auto [offset, count] = chunks[i];
        const auto initial_history = values(width * kernel, float(i) + 6);
        const auto initial_state = values(vh * kd * vd, float(i) + 8);
        histories.push_back(f32(context, { width, kernel }, initial_history));
        states.push_back(f32(context, { vh, kd, vd }, initial_state));
        auto history = f32(context, { width, kernel }, initial_history);
        auto state = f32(context, { vh, kd, vd }, initial_state);
        auto standalone_input = f32(context, { count, width },
                                    slice(input_data, offset * width, (offset + count) * width));
        auto standalone_a =
            f32(context, { count, vh }, slice(a_data, offset * vh, (offset + count) * vh));
        auto standalone_b =
            f32(context, { count, vh }, slice(b_data, offset * vh, (offset + count) * vh));
        auto standalone_conv = make_tensor(context, dtype::f32, { count, width });
        auto standalone_delta = make_tensor(context, dtype::f32, { count, value_width });
        REQUIRE(causal_conv1d_silu(context, standalone_input, weight, history, standalone_conv));
        REQUIRE(gated_delta_rule(context, standalone_conv, standalone_a, standalone_b, logs, bias,
                                 kh, state, standalone_delta));
        const auto conv_values = read_floats(standalone_conv);
        const auto delta_values = read_floats(standalone_delta);
        std::copy(conv_values.begin(), conv_values.end(), expected_conv.begin() + offset * width);
        std::copy(delta_values.begin(), delta_values.end(),
                  expected_delta.begin() + offset * value_width);
        expected_histories.push_back(read_floats(history));
        expected_states.push_back(read_floats(state));
    }

    compute_pass pass(context);
    REQUIRE(pass.begin());
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        REQUIRE(causal_conv1d_silu(context, input, weight, histories[i], conv, chunks[i]));
        REQUIRE(gated_delta_rule(context, conv, a, b, logs, bias, kh, states[i], delta, 1e-6F,
                                 chunks[i]));
    }
    REQUIRE(pass.finish());
    near(conv, expected_conv);
    near(delta, expected_delta);
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        CAPTURE(i);
        near(histories[i], expected_histories[i]);
        near(states[i], expected_states[i]);
    }
    const auto conv_values = read_floats(conv), delta_values = read_floats(delta);
    for (const std::size_t row : { 0, 3, 7 }) {
        CAPTURE(row);
        CHECK(slice(conv_values, row * width, (row + 1) * width)
              == std::vector<float>(width, sentinel));
        CHECK(slice(delta_values, row * value_width, (row + 1) * value_width)
              == std::vector<float>(value_width, sentinel));
    }
}

TEST_CASE("DeltaNet rejects empty, out-of-bounds, and overflow chunks without mutation")
{
    auto made = metal_context::make(load_shader_source());
    REQUIRE(made);
    auto& context = *made;
    constexpr std::size_t rows = 4, width = 8, vh = 2, kd = 2, vd = 2, kernel = 4;
    const auto initial_history = values(width * kernel, 1), initial_state = values(vh * kd * vd, 2);
    const std::vector<float> conv_sentinels(rows * width, -123.0F);
    const std::vector<float> delta_sentinels(rows * vh * vd, -456.0F);
    auto input = f32(context, { rows, width }, values(rows * width));
    auto weight = make_tensor(context, dtype::bf16, { width, 1, kernel });
    write_bf16(weight, values(width * kernel, 3));
    auto history = f32(context, { width, kernel }, initial_history);
    auto conv = f32(context, { rows, width }, conv_sentinels);
    auto a = f32(context, { rows, vh }, values(rows * vh, 4));
    auto b = f32(context, { rows, vh }, values(rows * vh, 5));
    auto logs = f32(context, { vh }, values(vh, 6));
    auto bias = make_tensor(context, dtype::bf16, { vh });
    write_bf16(bias, values(vh, 7));
    auto state = f32(context, { vh, kd, vd }, initial_state);
    auto delta = f32(context, { rows, vh * vd }, delta_sentinels);
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    const std::array chunks {
        deltanet_chunk { 0, 0 },     deltanet_chunk { rows, 0 },
        deltanet_chunk { rows, 1 },  deltanet_chunk { rows + 1, 1 },
        deltanet_chunk { 1, rows },  deltanet_chunk { 0, max },
        deltanet_chunk { 1, max },   deltanet_chunk { max, 1 },
        deltanet_chunk { max, max }, deltanet_chunk { max / sizeof(float) / width + 1, 1 },
    };
    compute_pass pass(context);
    REQUIRE(pass.begin());
    for (const auto chunk : chunks) {
        CAPTURE(chunk.offset);
        CAPTURE(chunk.count);
        const auto convolution = causal_conv1d_silu(context, input, weight, history, conv, chunk);
        REQUIRE_FALSE(convolution);
        CHECK(convolution.error() == tensor_op_errc::input_shape_mismatch);
        const auto recurrence =
            gated_delta_rule(context, input, a, b, logs, bias, 1, state, delta, 1e-6F, chunk);
        REQUIRE_FALSE(recurrence);
        CHECK(recurrence.error() == tensor_op_errc::input_shape_mismatch);

        // Raw dispatch must also reject ranges before computing overflowing byte offsets.
        const auto raw_conv = metal_kernels(context).dispatch_causal_conv1d_silu(
            input.buffer(), weight.buffer(), history.buffer(), conv.buffer(), chunk.count, width,
            kernel, chunk.offset);
        REQUIRE_FALSE(raw_conv);
        CHECK(raw_conv.error().code == metal_errc::invalid_input);
        const auto raw_delta = metal_kernels(context).dispatch_gated_delta_rule(
            input.buffer(), a.buffer(), b.buffer(), logs.buffer(), bias.buffer(), state.buffer(),
            delta.buffer(), chunk.count, 1, vh, kd, vd, 1e-6F, chunk.offset);
        REQUIRE_FALSE(raw_delta);
        CHECK(raw_delta.error().code == metal_errc::invalid_input);
    }
    REQUIRE(pass.finish());
    CHECK(read_floats(history) == initial_history);
    CHECK(read_floats(state) == initial_state);
    CHECK(read_floats(conv) == conv_sentinels);
    CHECK(read_floats(delta) == delta_sentinels);
}

TEST_CASE("DeltaNet raw dispatch bounds-checks every token buffer")
{
    auto made = metal_context::make(load_shader_source());
    REQUIRE(made);
    auto& context = *made;
    constexpr std::size_t rows = 4, width = 8, vh = 2, kd = 2, vd = 2, kernel = 4;
    const auto initial_history = values(width * kernel, 1), initial_state = values(vh * kd * vd, 2);
    const std::vector<float> conv_sentinels(rows * width, -123.0F);
    const std::vector<float> delta_sentinels(rows * vh * vd, -456.0F);
    auto input = f32(context, { rows, width }, values(rows * width));
    auto weight = make_tensor(context, dtype::bf16, { width, 1, kernel });
    write_bf16(weight, values(width * kernel, 3));
    auto history = f32(context, { width, kernel }, initial_history);
    auto conv = f32(context, { rows, width }, conv_sentinels);
    auto a = f32(context, { rows, vh }, values(rows * vh, 4));
    auto b = f32(context, { rows, vh }, values(rows * vh, 5));
    auto logs = f32(context, { vh }, values(vh, 6));
    auto bias = make_tensor(context, dtype::bf16, { vh });
    write_bf16(bias, values(vh, 7));
    auto state = f32(context, { vh, kd, vd }, initial_state);
    auto delta = f32(context, { rows, vh * vd }, delta_sentinels);
    auto short_input = f32(context, { rows - 1, width }, values((rows - 1) * width));
    auto short_gate = f32(context, { rows - 1, vh }, values((rows - 1) * vh));
    const std::vector<float> short_conv_sentinels((rows - 1) * width, -123.0F);
    const std::vector<float> short_delta_sentinels((rows - 1) * vh * vd, -456.0F);
    auto short_conv = f32(context, { rows - 1, width }, short_conv_sentinels);
    auto short_delta = f32(context, { rows - 1, vh * vd }, short_delta_sentinels);
    compute_pass pass(context);
    REQUIRE(pass.begin());
    // [2, 4) fits the full buffers but not a three-row buffer.
    for (int short_buffer = 0; short_buffer < 2; ++short_buffer) {
        CAPTURE(short_buffer);
        const auto dispatched = metal_kernels(context).dispatch_causal_conv1d_silu(
            short_buffer == 0 ? short_input.buffer() : input.buffer(), weight.buffer(),
            history.buffer(), short_buffer == 1 ? short_conv.buffer() : conv.buffer(), 2, width,
            kernel, 2);
        REQUIRE_FALSE(dispatched);
        CHECK(dispatched.error().code == metal_errc::invalid_input);
    }
    for (int short_buffer = 0; short_buffer < 4; ++short_buffer) {
        CAPTURE(short_buffer);
        const auto dispatched = metal_kernels(context).dispatch_gated_delta_rule(
            short_buffer == 0 ? short_input.buffer() : input.buffer(),
            short_buffer == 1 ? short_gate.buffer() : a.buffer(),
            short_buffer == 2 ? short_gate.buffer() : b.buffer(), logs.buffer(), bias.buffer(),
            state.buffer(), short_buffer == 3 ? short_delta.buffer() : delta.buffer(), 2, 1, vh, kd,
            vd, 1e-6F, 2);
        REQUIRE_FALSE(dispatched);
        CHECK(dispatched.error().code == metal_errc::invalid_input);
    }
    REQUIRE(pass.finish());
    CHECK(read_floats(history) == initial_history);
    CHECK(read_floats(state) == initial_state);
    CHECK(read_floats(conv) == conv_sentinels);
    CHECK(read_floats(delta) == delta_sentinels);
    CHECK(read_floats(short_conv) == short_conv_sentinels);
    CHECK(read_floats(short_delta) == short_delta_sentinels);
}

TEST_CASE("DeltaNet gated RMSNorm normalizes heads independently and uses direct f32 weights")
{
    const auto& context = test_context();
    auto input = f32(context, { 2, 6 }, { 0, 0, 0, 1, 2, 3, -1, 0, 1, 10, 20, 30 });
    auto gate = f32(context, { 2, 6 }, { 1, 1, 1, 0, 100, -100, 1, 2, 3, 0, 100, -100 });
    auto weight = f32(context, { 3 }, { 0, 1, 2 });
    auto output = make_tensor(context, dtype::f32, { 2, 6 });
    auto expected = read_floats(input), gates = read_floats(gate), weights = read_floats(weight);
    for (std::size_t base = 0; base < expected.size(); base += 3) {
        double square = 0;
        for (std::size_t j = 0; j < 3; ++j)
            square += double(expected[base + j]) * expected[base + j];
        for (std::size_t j = 0; j < 3; ++j)
            expected[base + j] = float(expected[base + j]
                                       / std::sqrt(square / 3 + 1e-6)
                                       * weights[j]
                                       * gates[base + j]
                                       / (1 + std::exp(-double(gates[base + j]))));
    }
    REQUIRE(rms_norm_gated(context, input, gate, weight, 1e-6F, output));
    near(output, expected);
}

TEST_CASE("DeltaNet rejects invalid geometry, dtypes, epsilon, and unsafe aliases before dispatch")
{
    const auto& c = test_context();
    auto input = make_tensor(c, dtype::f32, { 2, 6 });
    auto weight = make_tensor(c, dtype::bf16, { 6, 1, 4 });
    auto history = make_tensor(c, dtype::f32, { 6, 4 });
    auto output = make_tensor(c, dtype::f32, { 2, 6 });
    CHECK(causal_conv1d_silu(c, input, weight, history, input).error()
          == tensor_op_errc::unsupported_aliasing);
    auto bad_history = make_tensor(c, dtype::f32, { 5, 4 });
    CHECK(causal_conv1d_silu(c, input, weight, bad_history, output).error()
          == tensor_op_errc::input_shape_mismatch);
    auto bad_weight = make_tensor(c, dtype::f32, { 6, 1, 4 });
    CHECK(causal_conv1d_silu(c, input, bad_weight, history, output).error()
          == tensor_op_errc::unsupported_dtype);
    auto flat = make_tensor(c, dtype::f32, { 12 });
    CHECK(causal_conv1d_silu(c, flat, weight, history, output).error()
          == tensor_op_errc::invalid_rank);
    auto wrong_output = make_tensor(c, dtype::f32, { 1, 6 });
    CHECK(causal_conv1d_silu(c, input, weight, history, wrong_output).error()
          == tensor_op_errc::output_shape_mismatch);

    auto a = make_tensor(c, dtype::f32, { 2, 1 });
    auto logs = make_tensor(c, dtype::f32, { 1 });
    auto bias = make_tensor(c, dtype::bf16, { 1 });
    auto state = make_tensor(c, dtype::f32, { 1, 2, 2 });
    auto delta = make_tensor(c, dtype::f32, { 2, 2 });
    CHECK(gated_delta_rule(c, input, a, a, logs, bias, 0, state, delta).error()
          == tensor_op_errc::invalid_head_count);
    CHECK(gated_delta_rule(c, input, a, a, logs, bias, 2, state, delta).error()
          == tensor_op_errc::invalid_head_count);
    CHECK(gated_delta_rule(c, input, a, a, logs, logs, 1, state, delta).error()
          == tensor_op_errc::unsupported_dtype);
    CHECK(gated_delta_rule(c, input, input, a, logs, bias, 1, state, delta).error()
          == tensor_op_errc::input_shape_mismatch);
    CHECK(gated_delta_rule(c, input, a, a, logs, bias, 1, state, output).error()
          == tensor_op_errc::output_shape_mismatch);
    auto norm = make_tensor(c, dtype::f32, { 3 });
    CHECK(rms_norm_gated(c, input, input, norm, 1e-6F, input).error()
          == tensor_op_errc::unsupported_aliasing);
    CHECK(rms_norm_gated(c, input, input, norm, 1e-6F, delta).error()
          == tensor_op_errc::output_shape_mismatch);
    auto bad_norm = make_tensor(c, dtype::f32, { 4 });
    CHECK(rms_norm_gated(c, input, input, bad_norm, 1e-6F, output).error()
          == tensor_op_errc::input_shape_mismatch);
    for (float epsilon : { 0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::quiet_NaN() }) {
        CHECK(gated_delta_rule(c, input, a, a, logs, bias, 1, state, delta, epsilon).error()
              == tensor_op_errc::invalid_epsilon);
        CHECK(rms_norm_gated(c, input, input, norm, epsilon, output).error()
              == tensor_op_errc::invalid_epsilon);
    }
}
