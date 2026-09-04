#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "metal/metal_context.h"
#include "metal/metal_kv_cache.h"
#include "metal/metal_tensor.h"
#include "tensor/bf16.h"
#include "tensor/dtype.h"
#include "tensor/tensor_ops.h"

using chibillm::add;
using chibillm::bf16;
using chibillm::dtype;
using chibillm::embedding_lookup;
using chibillm::linear;
using chibillm::linear_add;
using chibillm::matmul;
using chibillm::metal_context;
using chibillm::metal_kv_cache;
using chibillm::metal_tensor;
using chibillm::paged_attention;
using chibillm::rms_norm;
using chibillm::rope;
using chibillm::silu_mul;
using chibillm::store_kv;
using chibillm::tensor_op_errc;

namespace {

std::string
load_shader_source()
{
    std::ifstream input(CHIBILLM_SHADER_PATH);
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
    };
}

metal_tensor
make_tensor(const metal_context& context, dtype type, std::vector<std::size_t> dimensions)
{
    auto tensor = metal_tensor::make(context, type, std::move(dimensions));
    REQUIRE(tensor.has_value());
    return std::move(*tensor);
}

void
write_floats(metal_tensor& tensor, const std::vector<float>& values)
{
    REQUIRE(tensor.descriptor().type() == dtype::f32);
    REQUIRE(tensor.buffer().size_bytes() == values.size() * sizeof(float));
    std::memcpy(tensor.buffer().bytes().data(), values.data(), tensor.buffer().size_bytes());
}

void
write_bf16(metal_tensor& tensor, const std::vector<float>& values)
{
    REQUIRE(tensor.descriptor().type() == dtype::bf16);
    REQUIRE(tensor.buffer().size_bytes() == values.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> bits;
    bits.reserve(values.size());
    for (const auto value : values) {
        bits.push_back(bf16::from_float(value).bits());
    }

    std::memcpy(tensor.buffer().bytes().data(), bits.data(), tensor.buffer().size_bytes());
}

void
write_i32(metal_tensor& tensor, const std::vector<std::int32_t>& values)
{
    REQUIRE(tensor.descriptor().type() == dtype::i32);
    REQUIRE(tensor.buffer().size_bytes() == values.size() * sizeof(std::int32_t));
    std::memcpy(tensor.buffer().bytes().data(), values.data(), tensor.buffer().size_bytes());
}

void
write_u32(metal_tensor& tensor, const std::vector<std::uint32_t>& values)
{
    REQUIRE(tensor.descriptor().type() == dtype::u32);
    REQUIRE(tensor.buffer().size_bytes() == values.size() * sizeof(std::uint32_t));
    std::memcpy(tensor.buffer().bytes().data(), values.data(), tensor.buffer().size_bytes());
}

std::vector<float>
read_floats(const metal_tensor& tensor)
{
    REQUIRE(tensor.descriptor().type() == dtype::f32);

    std::vector<float> values(tensor.descriptor().element_count());
    REQUIRE(tensor.buffer().size_bytes() == values.size() * sizeof(float));
    std::memcpy(values.data(), tensor.buffer().bytes().data(), tensor.buffer().size_bytes());
    return values;
}

metal_kv_cache
make_kv_cache(const metal_context& context)
{
    auto cache = metal_kv_cache::make(context,
                                      {
                                          .layer_count = 2,
                                          .block_count = 2,
                                          .block_size = 2,
                                          .kv_head_count = 1,
                                          .head_dimension = 2,
                                      });
    REQUIRE(cache.has_value());
    return std::move(*cache);
}

const metal_context&
test_context()
{
    static const auto context = [] {
        auto result = metal_context::make(load_shader_source());
        if (!result) {
            throw std::runtime_error("failed to create metal context: " + result.error().message);
        }
        return std::move(*result);
    }();
    return context;
}

} // namespace

TEST_CASE("matmul computes a rectangular matrix product")
{
    const auto& context = test_context();

    auto lhs = make_tensor(context, dtype::f32, { 2, 3 });
    auto rhs = make_tensor(context, dtype::f32, { 3, 4 });
    auto output = make_tensor(context, dtype::f32, { 2, 4 });
    write_floats(lhs, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
    write_floats(rhs,
                 { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F });

    auto multiplied = matmul(context, lhs, rhs, output);
    REQUIRE(multiplied.has_value());

    const auto values = read_floats(output);
    REQUIRE(values.size() == 8);
    CHECK(values[0] == doctest::Approx(38.0F));
    CHECK(values[1] == doctest::Approx(44.0F));
    CHECK(values[2] == doctest::Approx(50.0F));
    CHECK(values[3] == doctest::Approx(56.0F));
    CHECK(values[4] == doctest::Approx(83.0F));
    CHECK(values[5] == doctest::Approx(98.0F));
    CHECK(values[6] == doctest::Approx(113.0F));
    CHECK(values[7] == doctest::Approx(128.0F));
}

TEST_CASE("linear projects f32 input with row-major bf16 weights")
{
    const auto* context = &test_context();

    auto input = make_tensor(*context, dtype::f32, { 2, 3 });
    auto weight = make_tensor(*context, dtype::bf16, { 4, 3 });
    auto output = make_tensor(*context, dtype::f32, { 2, 4 });
    write_floats(input, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
    write_bf16(weight,
               { 1.0F, 5.0F, 9.0F, 2.0F, 6.0F, 10.0F, 3.0F, 7.0F, 11.0F, 4.0F, 8.0F, 12.0F });

    auto projected = linear(*context, input, weight, output);
    REQUIRE(projected.has_value());

    const auto values = read_floats(output);
    REQUIRE(values.size() == 8);
    CHECK(values[0] == doctest::Approx(38.0F));
    CHECK(values[1] == doctest::Approx(44.0F));
    CHECK(values[2] == doctest::Approx(50.0F));
    CHECK(values[3] == doctest::Approx(56.0F));
    CHECK(values[4] == doctest::Approx(83.0F));
    CHECK(values[5] == doctest::Approx(98.0F));
    CHECK(values[6] == doctest::Approx(113.0F));
    CHECK(values[7] == doctest::Approx(128.0F));
}

TEST_CASE("linear add fuses decode residuals and supports prefill rows")
{
    const auto* context = &test_context();

    auto weight = make_tensor(*context, dtype::bf16, { 4, 3 });
    write_bf16(weight,
               { 1.0F, 5.0F, 9.0F, 2.0F, 6.0F, 10.0F, 3.0F, 7.0F, 11.0F, 4.0F, 8.0F, 12.0F });

    auto decode_input = make_tensor(*context, dtype::f32, { 1, 3 });
    auto decode_residual = make_tensor(*context, dtype::f32, { 1, 4 });
    auto decode_output = make_tensor(*context, dtype::f32, { 1, 4 });
    write_floats(decode_input, { 1.0F, 2.0F, 3.0F });
    write_floats(decode_residual, { 1.0F, 2.0F, 3.0F, 4.0F });
    REQUIRE(linear_add(*context, decode_input, weight, decode_residual, decode_output).has_value());
    const auto decode_values = read_floats(decode_output);
    REQUIRE(decode_values.size() == 4);
    CHECK(decode_values[0] == doctest::Approx(39.0F));
    CHECK(decode_values[1] == doctest::Approx(46.0F));
    CHECK(decode_values[2] == doctest::Approx(53.0F));
    CHECK(decode_values[3] == doctest::Approx(60.0F));

    auto prefill_input = make_tensor(*context, dtype::f32, { 2, 3 });
    auto prefill_residual = make_tensor(*context, dtype::f32, { 2, 4 });
    auto prefill_output = make_tensor(*context, dtype::f32, { 2, 4 });
    write_floats(prefill_input, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
    write_floats(prefill_residual,
                 { 10.0F, 20.0F, 30.0F, 40.0F, 100.0F, 200.0F, 300.0F, 400.0F });
    REQUIRE(
        linear_add(*context, prefill_input, weight, prefill_residual, prefill_output).has_value());
    const auto prefill_values = read_floats(prefill_output);
    REQUIRE(prefill_values.size() == 8);
    CHECK(prefill_values[0] == doctest::Approx(48.0F));
    CHECK(prefill_values[1] == doctest::Approx(64.0F));
    CHECK(prefill_values[2] == doctest::Approx(80.0F));
    CHECK(prefill_values[3] == doctest::Approx(96.0F));
    CHECK(prefill_values[4] == doctest::Approx(183.0F));
    CHECK(prefill_values[5] == doctest::Approx(298.0F));
    CHECK(prefill_values[6] == doctest::Approx(413.0F));
    CHECK(prefill_values[7] == doctest::Approx(528.0F));
}

TEST_CASE("linear validates tensor ranks, dtypes, and shapes")
{
    const auto& context = test_context();

    auto valid_in = make_tensor(context, dtype::f32, { 2, 3 });
    auto valid_w = make_tensor(context, dtype::bf16, { 4, 3 });
    auto valid_out = make_tensor(context, dtype::f32, { 2, 4 });

    auto bad_rank = make_tensor(context, dtype::f32, { 6 });
    CHECK(linear(context, bad_rank, valid_w, valid_out).error() == tensor_op_errc::invalid_rank);

    auto bad_dtype = make_tensor(context, dtype::f32, { 4, 3 });
    CHECK(linear(context, valid_in, bad_dtype, valid_out).error() == tensor_op_errc::unsupported_dtype);

    auto bad_dim = make_tensor(context, dtype::bf16, { 4, 2 });
    CHECK(linear(context, valid_in, bad_dim, valid_out).error() == tensor_op_errc::inner_dimension_mismatch);

    auto bad_out = make_tensor(context, dtype::f32, { 2, 3 });
    CHECK(linear(context, valid_in, valid_w, bad_out).error() == tensor_op_errc::output_shape_mismatch);
}

TEST_CASE("linear preserves signed fractional bf16 weights")
{
    const auto* context = &test_context();

    auto input = make_tensor(*context, dtype::f32, { 1, 2 });
    auto weight = make_tensor(*context, dtype::bf16, { 2, 2 });
    auto output = make_tensor(*context, dtype::f32, { 1, 2 });
    write_floats(input, { 1.0F, 2.0F });
    write_bf16(weight, { 0.5F, -1.25F, -2.0F, 0.25F });

    auto projected = linear(*context, input, weight, output);
    REQUIRE(projected.has_value());

    const auto values = read_floats(output);
    REQUIRE(values.size() == 2);
    CHECK(values[0] == doctest::Approx(-2.0F));
    CHECK(values[1] == doctest::Approx(-1.5F));
}

TEST_CASE("linear decode handles a vectorized body and scalar tail")
{
    const auto* context = &test_context();

    auto input = make_tensor(*context, dtype::f32, { 1, 5 });
    auto weight = make_tensor(*context, dtype::bf16, { 2, 5 });
    auto output = make_tensor(*context, dtype::f32, { 1, 2 });
    write_floats(input, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F });
    write_bf16(weight, { 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, -1.0F, 2.0F, -3.0F, 4.0F, -5.0F });

    REQUIRE(linear(*context, input, weight, output).has_value());
    const auto values = read_floats(output);
    REQUIRE(values.size() == 2);
    CHECK(values[0] == doctest::Approx(15.0F));
    CHECK(values[1] == doctest::Approx(-15.0F));
}

TEST_CASE("linear split projects multiple rows into separate outputs")
{
    const auto* context = &test_context();

    auto input = make_tensor(*context, dtype::f32, { 2, 4 });
    auto weight = make_tensor(*context, dtype::bf16, { 5, 4 });
    auto output_a = make_tensor(*context, dtype::f32, { 2, 2 });
    auto output_b = make_tensor(*context, dtype::f32, { 2, 1 });
    auto output_c = make_tensor(*context, dtype::f32, { 2, 2 });
    write_floats(input, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F });
    write_bf16(weight,
               {
                   1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                   1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F,
               });

    auto projected = linear_split(*context, input, weight, { &output_a, &output_b, &output_c });
    REQUIRE(projected.has_value());

    CHECK(read_floats(output_a) == std::vector<float> { 1.0F, 2.0F, 5.0F, 6.0F });
    CHECK(read_floats(output_b) == std::vector<float> { 3.0F, 7.0F });
    CHECK(read_floats(output_c) == std::vector<float> { 4.0F, 10.0F, 8.0F, 26.0F });
}

TEST_CASE("embedding lookup gathers token rows in input order")
{
    const auto* context = &test_context();

    auto token_ids = make_tensor(*context, dtype::i32, { 2 });
    auto weight = make_tensor(*context, dtype::bf16, { 3, 4 });
    auto output = make_tensor(*context, dtype::f32, { 2, 4 });
    write_i32(token_ids, { 2, 0 });
    write_bf16(weight,
               { 0.5F, -1.25F, 2.0F, 0.0F, -2.0F, 0.25F, 1.5F, 3.0F, 4.0F, -0.5F, 0.75F, -3.0F });

    auto gathered = embedding_lookup(*context, token_ids, weight, output);
    REQUIRE(gathered.has_value());

    const auto values = read_floats(output);
    const std::vector<float> expected {
        4.0F, -0.5F, 0.75F, -3.0F, 0.5F, -1.25F, 2.0F, 0.0F,
    };
    REQUIRE(values.size() == expected.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        CHECK(values[index] == doctest::Approx(expected[index]));
    }
}

TEST_CASE("embedding lookup validates inputs and bounds")
{
    const auto& context = test_context();

    auto valid_ids = make_tensor(context, dtype::i32, { 2 });
    auto valid_w = make_tensor(context, dtype::bf16, { 3, 4 });
    auto valid_out = make_tensor(context, dtype::f32, { 2, 4 });

    auto bad_id_rank = make_tensor(context, dtype::i32, { 1, 2 });
    CHECK(embedding_lookup(context, bad_id_rank, valid_w, valid_out).error() == tensor_op_errc::invalid_rank);

    auto bad_id_dtype = make_tensor(context, dtype::f32, { 2 });
    CHECK(embedding_lookup(context, bad_id_dtype, valid_w, valid_out).error() == tensor_op_errc::unsupported_dtype);

    auto bad_w_dtype = make_tensor(context, dtype::f32, { 3, 4 });
    CHECK(embedding_lookup(context, valid_ids, bad_w_dtype, valid_out).error() == tensor_op_errc::unsupported_dtype);

    auto bad_out_shape = make_tensor(context, dtype::f32, { 1, 4 });
    CHECK(embedding_lookup(context, valid_ids, valid_w, bad_out_shape).error() == tensor_op_errc::output_shape_mismatch);

    write_i32(valid_ids, { 0, -1 });
    CHECK(embedding_lookup(context, valid_ids, valid_w, valid_out).error() == tensor_op_errc::token_out_of_range);

    write_i32(valid_ids, { 0, 3 });
    CHECK(embedding_lookup(context, valid_ids, valid_w, valid_out).error() == tensor_op_errc::token_out_of_range);
}

TEST_CASE("rms norm normalizes and scales each row independently")
{
    const auto* context = &test_context();

    auto input = make_tensor(*context, dtype::f32, { 2, 3 });
    auto weight = make_tensor(*context, dtype::bf16, { 3 });
    auto output = make_tensor(*context, dtype::f32, { 2, 3 });
    write_floats(input, { 1.0F, 2.0F, 2.0F, -1.0F, -2.0F, 2.0F });
    write_bf16(weight, { 1.0F, 0.5F, 2.0F });

    auto normalized = rms_norm(*context, input, weight, 1.0F, output);
    REQUIRE(normalized.has_value());

    const auto values = read_floats(output);
    const std::vector<float> expected { 0.5F, 0.5F, 2.0F, -0.5F, -0.5F, 2.0F };
    REQUIRE(values.size() == expected.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        CHECK(values[index] == doctest::Approx(expected[index]));
    }
}

TEST_CASE("rms norm validates inputs, shapes, and epsilon")
{
    const auto& context = test_context();

    auto valid_in = make_tensor(context, dtype::f32, { 2, 3 });
    auto valid_w = make_tensor(context, dtype::bf16, { 3 });
    auto valid_out = make_tensor(context, dtype::f32, { 2, 3 });

    auto bad_rank = make_tensor(context, dtype::f32, { 6 });
    CHECK(rms_norm(context, bad_rank, valid_w, 1.0F, valid_out).error() == tensor_op_errc::invalid_rank);

    auto bad_dtype = make_tensor(context, dtype::bf16, { 2, 3 });
    CHECK(rms_norm(context, bad_dtype, valid_w, 1.0F, valid_out).error() == tensor_op_errc::unsupported_dtype);

    auto bad_w_dim = make_tensor(context, dtype::bf16, { 4 });
    CHECK(rms_norm(context, valid_in, bad_w_dim, 1.0F, valid_out).error() == tensor_op_errc::inner_dimension_mismatch);

    auto bad_out_shape = make_tensor(context, dtype::f32, { 1, 3 });
    CHECK(rms_norm(context, valid_in, valid_w, 1.0F, bad_out_shape).error() == tensor_op_errc::output_shape_mismatch);

    for (const auto epsilon : { 0.0F, -1.0F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() }) {
        CHECK(rms_norm(context, valid_in, valid_w, epsilon, valid_out).error() == tensor_op_errc::invalid_epsilon);
    }
}

TEST_CASE("silu multiply gates the up projection elementwise")
{
    const auto* context = &test_context();

    auto gate = make_tensor(*context, dtype::f32, { 2, 2 });
    auto up = make_tensor(*context, dtype::f32, { 2, 2 });
    auto output = make_tensor(*context, dtype::f32, { 2, 2 });
    write_floats(gate, { 0.0F, 1.0F, -1.0F, 2.0F });
    write_floats(up, { 2.0F, 3.0F, 4.0F, -0.5F });

    auto activated = silu_mul(*context, gate, up, output);
    REQUIRE(activated.has_value());

    const auto values = read_floats(output);
    const std::vector<float> expected { 0.0F, 2.1931758F, -1.0757657F, -0.8807971F };
    REQUIRE(values.size() == expected.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        CHECK(values[index] == doctest::Approx(expected[index]));
    }
}

TEST_CASE("silu multiply validates shapes and dtypes")
{
    const auto& context = test_context();
    auto valid = make_tensor(context, dtype::f32, { 2, 2 });
    auto bad_rank = make_tensor(context, dtype::f32, { 4 });
    auto bad_dtype = make_tensor(context, dtype::bf16, { 2, 2 });
    auto bad_shape = make_tensor(context, dtype::f32, { 1, 4 });

    CHECK(silu_mul(context, bad_rank, valid, valid).error() == tensor_op_errc::invalid_rank);
    CHECK(silu_mul(context, bad_dtype, valid, valid).error() == tensor_op_errc::unsupported_dtype);
    CHECK(silu_mul(context, valid, bad_shape, valid).error() == tensor_op_errc::input_shape_mismatch);
    CHECK(silu_mul(context, valid, valid, bad_shape).error() == tensor_op_errc::output_shape_mismatch);
}

TEST_CASE("add sums corresponding tensor elements")
{
    const auto& context = test_context();

    auto lhs = make_tensor(context, dtype::f32, { 2, 3 });
    auto rhs = make_tensor(context, dtype::f32, { 2, 3 });
    auto output = make_tensor(context, dtype::f32, { 2, 3 });
    write_floats(lhs, { 1.0F, -2.0F, 0.5F, 4.0F, -1.25F, 10.0F });
    write_floats(rhs, { 2.0F, 3.0F, -0.5F, -4.0F, 1.0F, -2.0F });

    auto added = add(context, lhs, rhs, output);
    REQUIRE(added.has_value());

    const auto values = read_floats(output);
    const std::vector<float> expected { 3.0F, 1.0F, 0.0F, 0.0F, -0.25F, 8.0F };
    REQUIRE(values.size() == expected.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        CHECK(values[index] == doctest::Approx(expected[index]));
    }
}

TEST_CASE("add validates shapes and dtypes")
{
    const auto& context = test_context();
    auto valid = make_tensor(context, dtype::f32, { 2, 2 });
    auto bad_rank = make_tensor(context, dtype::f32, { 4 });
    auto bad_dtype = make_tensor(context, dtype::bf16, { 2, 2 });
    auto bad_shape = make_tensor(context, dtype::f32, { 1, 4 });

    CHECK(add(context, bad_rank, valid, valid).error() == tensor_op_errc::invalid_rank);
    CHECK(add(context, bad_dtype, valid, valid).error() == tensor_op_errc::unsupported_dtype);
    CHECK(add(context, valid, bad_shape, valid).error() == tensor_op_errc::input_shape_mismatch);
    CHECK(add(context, valid, valid, bad_shape).error() == tensor_op_errc::output_shape_mismatch);
}

TEST_CASE("rope rotates split-half feature pairs for every head")
{
    const auto* context = &test_context();

    auto input = make_tensor(*context, dtype::f32, { 2, 8 });
    auto positions = make_tensor(*context, dtype::u32, { 2 });
    auto output = make_tensor(*context, dtype::f32, { 2, 8 });
    write_floats(input,
                 { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                   6.0F, 7.0F, 8.0F });
    write_u32(positions, { 0, 1 });

    auto rotated = rope(*context, input, positions, 2, 10'000.0F, output);
    REQUIRE(rotated.has_value());

    const auto full_cosine = std::cos(1.0F);
    const auto full_sine = std::sin(1.0F);
    const auto slow_cosine = std::cos(0.01F);
    const auto slow_sine = std::sin(0.01F);
    const std::vector<float> expected {
        1.0F,
        2.0F,
        3.0F,
        4.0F,
        5.0F,
        6.0F,
        7.0F,
        8.0F,
        1.0F * full_cosine - 3.0F * full_sine,
        2.0F * slow_cosine - 4.0F * slow_sine,
        3.0F * full_cosine + 1.0F * full_sine,
        4.0F * slow_cosine + 2.0F * slow_sine,
        5.0F * full_cosine - 7.0F * full_sine,
        6.0F * slow_cosine - 8.0F * slow_sine,
        7.0F * full_cosine + 5.0F * full_sine,
        8.0F * slow_cosine + 6.0F * slow_sine,
    };

    const auto values = read_floats(output);
    REQUIRE(values.size() == expected.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        CHECK(values[index] == doctest::Approx(expected[index]).epsilon(1e-5));
    }
}

TEST_CASE("rope validates inputs, shapes, and theta")
{
    const auto& context = test_context();

    SUBCASE("ranks and dtypes")
    {
        auto input = make_tensor(context, dtype::f32, { 2, 4 });
        auto positions = make_tensor(context, dtype::u32, { 2 });
        auto output = make_tensor(context, dtype::f32, { 2, 4 });

        CHECK(rope(context, make_tensor(context, dtype::f32, { 8 }), positions, 1, 10'000.0F, output).error()
              == tensor_op_errc::invalid_rank);
        CHECK(rope(context, input, make_tensor(context, dtype::u32, { 1, 2 }), 1, 10'000.0F, output).error()
              == tensor_op_errc::invalid_rank);
        auto bad_rank_out = make_tensor(context, dtype::f32, { 8 });
        CHECK(rope(context, input, positions, 1, 10'000.0F, bad_rank_out).error()
              == tensor_op_errc::invalid_rank);

        CHECK(rope(context, make_tensor(context, dtype::bf16, { 2, 4 }), positions, 1, 10'000.0F, output).error()
              == tensor_op_errc::unsupported_dtype);
        CHECK(rope(context, input, make_tensor(context, dtype::i32, { 2 }), 1, 10'000.0F, output).error()
              == tensor_op_errc::unsupported_dtype);
        auto bad_dtype_out = make_tensor(context, dtype::bf16, { 2, 4 });
        CHECK(rope(context, input, positions, 1, 10'000.0F, bad_dtype_out).error()
              == tensor_op_errc::unsupported_dtype);
    }

    SUBCASE("shapes and head count")
    {
        auto input = make_tensor(context, dtype::f32, { 2, 8 });
        auto positions = make_tensor(context, dtype::u32, { 2 });
        auto output = make_tensor(context, dtype::f32, { 2, 8 });

        CHECK(rope(context, input, make_tensor(context, dtype::u32, { 1 }), 1, 10'000.0F, output).error()
              == tensor_op_errc::position_count_mismatch);
        CHECK(rope(context, input, positions, 0, 10'000.0F, output).error()
              == tensor_op_errc::invalid_head_count);
        CHECK(rope(context, input, positions, 3, 10'000.0F, output).error()
              == tensor_op_errc::invalid_head_dimension);
        auto six_dim_tensor = make_tensor(context, dtype::f32, { 2, 6 });
        CHECK(rope(context, six_dim_tensor, positions, 2, 10'000.0F, six_dim_tensor).error()
              == tensor_op_errc::invalid_head_dimension);
        auto bad_shape_out = make_tensor(context, dtype::f32, { 1, 16 });
        CHECK(rope(context, input, positions, 2, 10'000.0F, bad_shape_out).error()
              == tensor_op_errc::output_shape_mismatch);
    }

    SUBCASE("theta validation")
    {
        auto input = make_tensor(context, dtype::f32, { 2, 8 });
        auto positions = make_tensor(context, dtype::u32, { 2 });
        auto output = make_tensor(context, dtype::f32, { 2, 8 });

        for (const auto theta : { 0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::quiet_NaN() }) {
            CHECK(rope(context, input, positions, 2, theta, output).error()
                  == tensor_op_errc::invalid_rope_theta);
        }
    }
}

TEST_CASE("kv cache store writes rows into selected physical slots")
{
    const auto& context = test_context();
    auto cache = make_kv_cache(context);

    auto keys = make_tensor(context, dtype::f32, { 2, 2 });
    auto values = make_tensor(context, dtype::f32, { 2, 2 });
    auto slots = make_tensor(context, dtype::u32, { 2 });
    write_floats(keys, { 1.0F, 2.0F, 3.0F, 4.0F });
    write_floats(values, { 5.0F, 6.0F, 7.0F, 8.0F });
    write_u32(slots, { 1, 3 });
    write_floats(cache.keys(), std::vector<float>(16, -1.0F));
    write_floats(cache.values(), std::vector<float>(16, -1.0F));

    auto stored = store_kv(context, keys, values, slots, 1, cache);
    REQUIRE(stored.has_value());

    CHECK(read_floats(cache.keys())
          == std::vector<float> { -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F,
                                  -1.0F, 1.0F, 2.0F, -1.0F, -1.0F, 3.0F, 4.0F });
    CHECK(read_floats(cache.values())
          == std::vector<float> { -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F, -1.0F,
                                  -1.0F, 5.0F, 6.0F, -1.0F, -1.0F, 7.0F, 8.0F });
}

TEST_CASE("kv cache store validates inputs, shapes, and slots")
{
    const auto& context = test_context();
    auto cache = make_kv_cache(context);
    auto keys = make_tensor(context, dtype::f32, { 2, 2 });
    auto values = make_tensor(context, dtype::f32, { 2, 2 });
    auto slots = make_tensor(context, dtype::u32, { 2 });

    SUBCASE("ranks and dtypes")
    {
        CHECK(store_kv(context, make_tensor(context, dtype::f32, { 4 }), values, slots, 0, cache).error()
              == tensor_op_errc::invalid_rank);
        CHECK(store_kv(context, keys, make_tensor(context, dtype::f32, { 4 }), slots, 0, cache).error()
              == tensor_op_errc::invalid_rank);
        CHECK(store_kv(context, keys, values, make_tensor(context, dtype::u32, { 1, 2 }), 0, cache).error()
              == tensor_op_errc::invalid_rank);

        CHECK(store_kv(context, make_tensor(context, dtype::bf16, { 2, 2 }), values, slots, 0, cache).error()
              == tensor_op_errc::unsupported_dtype);
        CHECK(store_kv(context, keys, make_tensor(context, dtype::bf16, { 2, 2 }), slots, 0, cache).error()
              == tensor_op_errc::unsupported_dtype);
        CHECK(store_kv(context, keys, values, make_tensor(context, dtype::i32, { 2 }), 0, cache).error()
              == tensor_op_errc::unsupported_dtype);
    }

    SUBCASE("shapes, layers, and slot bounds")
    {
        // mismatched key/value shapes
        CHECK(store_kv(context, keys, make_tensor(context, dtype::f32, { 1, 2 }), slots, 0, cache).error()
              == tensor_op_errc::input_shape_mismatch);
        // slot count mismatch
        CHECK(store_kv(context, keys, values, make_tensor(context, dtype::u32, { 1 }), 0, cache).error()
              == tensor_op_errc::cache_slot_count_mismatch);
        // feature dimension mismatch
        CHECK(store_kv(context, make_tensor(context, dtype::f32, { 2, 3 }),
                       make_tensor(context, dtype::f32, { 2, 3 }), slots, 0, cache).error()
              == tensor_op_errc::cache_feature_count_mismatch);
        // layer out of range
        CHECK(store_kv(context, keys, values, slots, 2, cache).error()
              == tensor_op_errc::cache_layer_out_of_range);
        // slot out of range
        auto invalid_slots = make_tensor(context, dtype::u32, { 2 });
        write_u32(invalid_slots, { 0, 4 });
        CHECK(store_kv(context, keys, values, invalid_slots, 0, cache).error()
              == tensor_op_errc::cache_slot_out_of_range);
    }
}

TEST_CASE("paged attention follows block tables and shares kv heads")
{
    const auto* context = &test_context();
    auto cache = make_kv_cache(*context);

    write_floats(cache.keys(),
                 { 1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                   0.0F, 0.0F, 0.0F });
    write_floats(cache.values(),
                 { 30.0F, 3.0F, 0.0F, 0.0F, 10.0F, 1.0F, 20.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                   0.0F, 0.0F, 0.0F });

    auto queries = make_tensor(*context, dtype::f32, { 1, 4 });
    auto positions = make_tensor(*context, dtype::u32, { 1 });
    auto block_table = make_tensor(*context, dtype::u32, { 2 });
    auto table_offsets = make_tensor(*context, dtype::u32, { 1 });
    auto table_lengths = make_tensor(*context, dtype::u32, { 1 });
    auto output = make_tensor(*context, dtype::f32, { 1, 4 });
    write_floats(queries, { 1.0F, 0.0F, 0.0F, 1.0F });
    write_u32(positions, { 2 });
    write_u32(block_table, { 1, 0 });
    write_u32(table_offsets, { 0 });
    write_u32(table_lengths, { 2 });

    auto attended = paged_attention(*context, queries, positions, block_table, table_offsets,
                                    table_lengths, 0, 2, cache, output);
    REQUIRE(attended.has_value());

    const auto exponential = std::exp(1.0F / std::sqrt(2.0F));
    const auto denominator = 2.0F * exponential + 1.0F;
    const std::vector<float> expected {
        (40.0F * exponential + 20.0F) / denominator,
        (4.0F * exponential + 2.0F) / denominator,
        (10.0F + 50.0F * exponential) / denominator,
        (1.0F + 5.0F * exponential) / denominator,
    };

    const auto result = read_floats(output);
    REQUIRE(result.size() == expected.size());
    for (std::size_t index = 0; index < result.size(); ++index) {
        CHECK(result[index] == doctest::Approx(expected[index]).epsilon(1e-5));
    }
}

TEST_CASE("paged FlashAttention tiles causal query rows over paged keys and values")
{
    const auto* context = &test_context();
    auto cache_result = metal_kv_cache::make(*context,
                                             {
                                                 .layer_count = 1,
                                                 .block_count = 2,
                                                 .block_size = 2,
                                                 .kv_head_count = 1,
                                                 .head_dimension = 2,
                                             });
    REQUIRE(cache_result.has_value());
    auto cache = std::move(*cache_result);

    // Logical tokens 0..3 are physically paged as block 1 followed by block 0.
    const std::vector<float> key_values {
        0.25F, 0.75F, -0.5F, 1.0F, 1.0F, 0.0F, 0.5F, -0.25F,
    };
    const std::vector<float> value_values {
        30.0F, 3.0F, 40.0F, 4.0F, 10.0F, 1.0F, 20.0F, 2.0F,
    };
    write_floats(cache.keys(), key_values);
    write_floats(cache.values(), value_values);

    auto queries = make_tensor(*context, dtype::f32, { 4, 4 });
    auto positions = make_tensor(*context, dtype::u32, { 4 });
    auto block_table = make_tensor(*context, dtype::u32, { 2 });
    auto table_offsets = make_tensor(*context, dtype::u32, { 4 });
    auto table_lengths = make_tensor(*context, dtype::u32, { 4 });
    auto output = make_tensor(*context, dtype::f32, { 4, 4 });
    const std::vector<float> query_values {
        1.0F,  0.5F,  -0.5F, 1.0F, 0.25F, 1.0F,  1.0F,   -0.25F,
        -0.5F, 0.75F, 0.5F,  0.5F, 1.0F,  -1.0F, -0.25F, 0.75F,
    };
    write_floats(queries, query_values);
    write_u32(positions, { 0, 1, 2, 3 });
    write_u32(block_table, { 1, 0 });
    write_u32(table_offsets, { 0, 0, 0, 0 });
    write_u32(table_lengths, { 2, 2, 2, 2 });

    REQUIRE(paged_attention(*context, queries, positions, block_table, table_offsets, table_lengths,
                            0, 2, cache, output)
                .has_value());

    std::vector<float> expected(query_values.size(), 0.0F);
    constexpr std::array<std::uint32_t, 2> physical_blocks { 1, 0 };
    const float scale = 1.0F / std::sqrt(2.0F);
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t head = 0; head < 2; ++head) {
            std::vector<float> scores(row + 1);
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t token = 0; token <= row; ++token) {
                const auto block = physical_blocks[token / 2];
                const auto offset = token % 2;
                float score = 0.0F;
                for (std::size_t feature = 0; feature < 2; ++feature) {
                    const auto cache_index = *cache.element_offset(0, block, offset, 0, feature);
                    score += query_values[row * 4 + head * 2 + feature] * key_values[cache_index];
                }
                scores[token] = score * scale;
                maximum = std::max(maximum, scores[token]);
            }

            float denominator = 0.0F;
            for (std::size_t token = 0; token <= row; ++token) {
                const auto weight = std::exp(scores[token] - maximum);
                denominator += weight;
                const auto block = physical_blocks[token / 2];
                const auto offset = token % 2;
                for (std::size_t feature = 0; feature < 2; ++feature) {
                    const auto cache_index = *cache.element_offset(0, block, offset, 0, feature);
                    expected[row * 4 + head * 2 + feature] += weight * value_values[cache_index];
                }
            }
            for (std::size_t feature = 0; feature < 2; ++feature) {
                expected[row * 4 + head * 2 + feature] /= denominator;
            }
        }
    }

    const auto actual = read_floats(output);
    REQUIRE(actual.size() == expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        CHECK(actual[index] == doctest::Approx(expected[index]).epsilon(1e-5));
    }
}

TEST_CASE("paged FlashAttention merges key tiles across a cached prefix")
{
    const auto* context = &test_context();
    auto cache_result = metal_kv_cache::make(*context,
                                             {
                                                 .layer_count = 1,
                                                 .block_count = 2,
                                                 .block_size = 16,
                                                 .kv_head_count = 1,
                                                 .head_dimension = 2,
                                             });
    REQUIRE(cache_result.has_value());
    auto cache = std::move(*cache_result);

    write_floats(cache.keys(), std::vector<float>(cache.element_count(), 0.0F));
    std::vector<float> values;
    values.reserve(cache.element_count());
    for (std::size_t token = 0; token < 32; ++token) {
        values.push_back(static_cast<float>(token));
        values.push_back(static_cast<float>(2 * token));
    }
    write_floats(cache.values(), values);

    auto queries = make_tensor(*context, dtype::f32, { 8, 2 });
    auto positions = make_tensor(*context, dtype::u32, { 8 });
    auto block_table = make_tensor(*context, dtype::u32, { 2 });
    auto table_offsets = make_tensor(*context, dtype::u32, { 8 });
    auto table_lengths = make_tensor(*context, dtype::u32, { 8 });
    auto output = make_tensor(*context, dtype::f32, { 8, 2 });
    write_floats(queries, std::vector<float>(16, 1.0F));
    write_u32(positions, { 16, 17, 18, 19, 20, 21, 22, 23 });
    write_u32(block_table, { 0, 1 });
    write_u32(table_offsets, std::vector<std::uint32_t>(8, 0));
    write_u32(table_lengths, std::vector<std::uint32_t>(8, 2));

    REQUIRE(paged_attention(*context, queries, positions, block_table, table_offsets, table_lengths,
                            0, 1, cache, output)
                .has_value());

    const auto actual = read_floats(output);
    REQUIRE(actual.size() == 16);
    for (std::size_t row = 0; row < 8; ++row) {
        const auto position = static_cast<float>(16 + row);
        CHECK(actual[row * 2] == doctest::Approx(position / 2.0F).epsilon(1e-5));
        CHECK(actual[row * 2 + 1] == doctest::Approx(position).epsilon(1e-5));
    }
}

TEST_CASE("paged attention combines multiple context chunks")
{
    const auto* context = &test_context();
    auto cache_result = metal_kv_cache::make(*context,
                                             {
                                                 .layer_count = 1,
                                                 .block_count = 9,
                                                 .block_size = 16,
                                                 .kv_head_count = 1,
                                                 .head_dimension = 2,
                                             });
    REQUIRE(cache_result.has_value());
    auto cache = std::move(*cache_result);

    write_floats(cache.keys(), std::vector<float>(cache.element_count(), 0.0F));
    std::vector<float> values;
    values.reserve(cache.element_count());
    for (std::size_t token = 0; token < 144; ++token) {
        values.push_back(static_cast<float>(token));
        values.push_back(static_cast<float>(token * 2));
    }
    write_floats(cache.values(), values);

    auto queries = make_tensor(*context, dtype::f32, { 1, 2 });
    auto positions = make_tensor(*context, dtype::u32, { 1 });
    auto block_table = make_tensor(*context, dtype::u32, { 9 });
    auto table_offsets = make_tensor(*context, dtype::u32, { 1 });
    auto table_lengths = make_tensor(*context, dtype::u32, { 1 });
    auto output = make_tensor(*context, dtype::f32, { 1, 2 });
    write_floats(queries, { 1.0F, 1.0F });
    write_u32(positions, { 128 });
    write_u32(block_table, { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
    write_u32(table_offsets, { 0 });
    write_u32(table_lengths, { 9 });

    auto attended = paged_attention(*context, queries, positions, block_table, table_offsets,
                                    table_lengths, 0, 1, cache, output);
    REQUIRE(attended.has_value());

    const auto result = read_floats(output);
    REQUIRE(result.size() == 2);
    CHECK(result[0] == doctest::Approx(64.0F).epsilon(1e-5));
    CHECK(result[1] == doctest::Approx(128.0F).epsilon(1e-5));
}

TEST_CASE("paged attention validates inputs and metadata")
{
    const auto& context = test_context();
    auto cache = make_kv_cache(context);
    auto queries = make_tensor(context, dtype::f32, { 1, 2 });
    auto positions = make_tensor(context, dtype::u32, { 1 });
    auto table = make_tensor(context, dtype::u32, { 1 });
    auto offsets = make_tensor(context, dtype::u32, { 1 });
    auto lengths = make_tensor(context, dtype::u32, { 1 });
    auto output = make_tensor(context, dtype::f32, { 1, 2 });

    SUBCASE("ranks and dtypes")
    {
        CHECK(paged_attention(context, make_tensor(context, dtype::f32, { 2 }), positions, table, offsets,
                              lengths, 0, 1, cache, output)
                  .error()
              == tensor_op_errc::invalid_rank);
        CHECK(paged_attention(context, queries, positions, make_tensor(context, dtype::i32, { 1 }), offsets,
                              lengths, 0, 1, cache, output)
                  .error()
              == tensor_op_errc::unsupported_dtype);
    }

    SUBCASE("metadata counts")
    {
        auto q2 = make_tensor(context, dtype::f32, { 2, 2 });
        auto out2 = make_tensor(context, dtype::f32, { 2, 2 });
        CHECK(paged_attention(context, q2, positions, table, make_tensor(context, dtype::u32, { 2 }),
                              make_tensor(context, dtype::u32, { 2 }), 0, 1, cache, out2)
                  .error()
              == tensor_op_errc::position_count_mismatch);
        CHECK(paged_attention(context, q2, make_tensor(context, dtype::u32, { 2 }), table, offsets,
                              make_tensor(context, dtype::u32, { 2 }), 0, 1, cache, out2)
                  .error()
              == tensor_op_errc::block_table_metadata_count_mismatch);
    }

    SUBCASE("heads and dimensions")
    {
        CHECK(paged_attention(context, queries, positions, table, offsets, lengths, 0, 0, cache, output)
                  .error()
              == tensor_op_errc::invalid_head_count);
        auto dim5 = make_tensor(context, dtype::f32, { 1, 5 });
        CHECK(paged_attention(context, dim5, positions, table, offsets, lengths, 0, 2, cache, dim5)
                  .error()
              == tensor_op_errc::invalid_head_dimension);
        auto dim3 = make_tensor(context, dtype::f32, { 1, 3 });
        CHECK(paged_attention(context, dim3, positions, table, offsets, lengths, 0, 1, cache, dim3)
                  .error()
              == tensor_op_errc::cache_head_dimension_mismatch);

        auto gqa_cache = metal_kv_cache::make(context, { .layer_count = 1,
                                                          .block_count = 1,
                                                          .block_size = 1,
                                                          .kv_head_count = 2,
                                                          .head_dimension = 2 });
        REQUIRE(gqa_cache.has_value());
        auto dim6 = make_tensor(context, dtype::f32, { 1, 6 });
        CHECK(paged_attention(context, dim6, positions, table, offsets, lengths, 0, 3, *gqa_cache, dim6)
                  .error()
              == tensor_op_errc::invalid_kv_head_mapping);
    }

    SUBCASE("output and layer")
    {
        auto bad_out = make_tensor(context, dtype::f32, { 2, 1 });
        CHECK(paged_attention(context, queries, positions, table, offsets, lengths, 0, 1, cache, bad_out)
                  .error()
              == tensor_op_errc::output_shape_mismatch);
        CHECK(paged_attention(context, queries, positions, table, offsets, lengths, 2, 1, cache, output)
                  .error()
              == tensor_op_errc::cache_layer_out_of_range);
    }

    SUBCASE("block table bounds and ids")
    {
        // missing required block
        write_u32(positions, { 2 });
        write_u32(offsets, { 0 });
        write_u32(lengths, { 1 });
        write_u32(table, { 0 });
        CHECK(paged_attention(context, queries, positions, table, offsets, lengths, 0, 1, cache, output)
                  .error()
              == tensor_op_errc::block_table_range_out_of_bounds);

        // range outside flattened table
        write_u32(positions, { 0 });
        write_u32(offsets, { 1 });
        write_u32(lengths, { 2 });
        auto t2 = make_tensor(context, dtype::u32, { 2 });
        write_u32(t2, { 0, 1 });
        CHECK(paged_attention(context, queries, positions, t2, offsets, lengths, 0, 1, cache, output)
                  .error()
              == tensor_op_errc::block_table_range_out_of_bounds);

        // physical block out of range
        write_u32(positions, { 0 });
        write_u32(offsets, { 0 });
        write_u32(lengths, { 1 });
        write_u32(table, { 2 });
        CHECK(paged_attention(context, queries, positions, table, offsets, lengths, 0, 1, cache, output)
                  .error()
              == tensor_op_errc::cache_block_out_of_range);
    }
}
