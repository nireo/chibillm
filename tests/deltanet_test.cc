#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "metal_test_support.h"
#include "tensor/deltanet.h"
#include <algorithm>
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
            const double decay = std::exp(-std::exp(double(logs[h])) * std::log1p(std::exp(x)));
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
