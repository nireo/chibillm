#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "metal/metal_context.h"
#include "metal/metal_tensor.h"
#include "tensor/bf16.h"
#include "tensor/dtype.h"
#include "tensor/tensor_descriptor.h"
#include "tensor/tensor_ops.h"
#include "tensor/tensor_shape.h"

using chibillm::bf16;
using chibillm::dtype;
using chibillm::embedding_lookup;
using chibillm::linear;
using chibillm::matmul;
using chibillm::metal_context;
using chibillm::metal_tensor;
using chibillm::rms_norm;
using chibillm::silu_mul;
using chibillm::tensor_descriptor;
using chibillm::tensor_op_errc;
using chibillm::tensor_shape;

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
    auto shape = tensor_shape::make(std::move(dimensions));
    REQUIRE(shape.has_value());

    auto descriptor = tensor_descriptor::make(type, std::move(*shape));
    REQUIRE(descriptor.has_value());

    auto tensor = metal_tensor::make(context, std::move(*descriptor));
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

std::vector<float>
read_floats(const metal_tensor& tensor)
{
    REQUIRE(tensor.descriptor().type() == dtype::f32);

    std::vector<float> values(tensor.descriptor().element_count());
    REQUIRE(tensor.buffer().size_bytes() == values.size() * sizeof(float));
    std::memcpy(values.data(), tensor.buffer().bytes().data(), tensor.buffer().size_bytes());
    return values;
}

} // namespace

TEST_CASE("matmul computes a rectangular matrix product")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto lhs = make_tensor(*context, dtype::f32, { 2, 3 });
    auto rhs = make_tensor(*context, dtype::f32, { 3, 4 });
    auto output = make_tensor(*context, dtype::f32, { 2, 4 });
    write_floats(lhs, { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F });
    write_floats(rhs,
                 { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F });

    auto multiplied = matmul(*context, lhs, rhs, output);
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

TEST_CASE("matmul requires rank-two tensors")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto lhs = make_tensor(*context, dtype::f32, { 6 });
    auto rhs = make_tensor(*context, dtype::f32, { 3, 2 });
    auto output = make_tensor(*context, dtype::f32, { 2, 2 });

    auto multiplied = matmul(*context, lhs, rhs, output);
    REQUIRE_FALSE(multiplied.has_value());
    CHECK(multiplied.error() == tensor_op_errc::invalid_rank);
}

TEST_CASE("matmul initially supports only f32 tensors")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto lhs = make_tensor(*context, dtype::bf16, { 2, 3 });
    auto rhs = make_tensor(*context, dtype::f32, { 3, 2 });
    auto output = make_tensor(*context, dtype::f32, { 2, 2 });

    auto multiplied = matmul(*context, lhs, rhs, output);
    REQUIRE_FALSE(multiplied.has_value());
    CHECK(multiplied.error() == tensor_op_errc::unsupported_dtype);
}

TEST_CASE("matmul requires matching inner dimensions")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto lhs = make_tensor(*context, dtype::f32, { 2, 3 });
    auto rhs = make_tensor(*context, dtype::f32, { 4, 2 });
    auto output = make_tensor(*context, dtype::f32, { 2, 2 });

    auto multiplied = matmul(*context, lhs, rhs, output);
    REQUIRE_FALSE(multiplied.has_value());
    CHECK(multiplied.error() == tensor_op_errc::inner_dimension_mismatch);
}

TEST_CASE("matmul requires the exact output shape")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto lhs = make_tensor(*context, dtype::f32, { 2, 3 });
    auto rhs = make_tensor(*context, dtype::f32, { 3, 2 });
    auto output = make_tensor(*context, dtype::f32, { 2, 3 });

    auto multiplied = matmul(*context, lhs, rhs, output);
    REQUIRE_FALSE(multiplied.has_value());
    CHECK(multiplied.error() == tensor_op_errc::output_shape_mismatch);
}

TEST_CASE("linear projects f32 input with row-major bf16 weights")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

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

TEST_CASE("linear requires rank-two tensors")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto input = make_tensor(*context, dtype::f32, { 6 });
    auto weight = make_tensor(*context, dtype::bf16, { 4, 3 });
    auto output = make_tensor(*context, dtype::f32, { 2, 4 });

    auto projected = linear(*context, input, weight, output);
    REQUIRE_FALSE(projected.has_value());
    CHECK(projected.error() == tensor_op_errc::invalid_rank);
}

TEST_CASE("linear requires f32 input and output with bf16 weights")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto input = make_tensor(*context, dtype::f32, { 2, 3 });
    auto weight = make_tensor(*context, dtype::f32, { 4, 3 });
    auto output = make_tensor(*context, dtype::f32, { 2, 4 });

    auto projected = linear(*context, input, weight, output);
    REQUIRE_FALSE(projected.has_value());
    CHECK(projected.error() == tensor_op_errc::unsupported_dtype);
}

TEST_CASE("linear requires matching input feature dimensions")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto input = make_tensor(*context, dtype::f32, { 2, 3 });
    auto weight = make_tensor(*context, dtype::bf16, { 4, 2 });
    auto output = make_tensor(*context, dtype::f32, { 2, 4 });

    auto projected = linear(*context, input, weight, output);
    REQUIRE_FALSE(projected.has_value());
    CHECK(projected.error() == tensor_op_errc::inner_dimension_mismatch);
}

TEST_CASE("linear requires output to match rows and output features")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto input = make_tensor(*context, dtype::f32, { 2, 3 });
    auto weight = make_tensor(*context, dtype::bf16, { 4, 3 });
    auto output = make_tensor(*context, dtype::f32, { 2, 3 });

    auto projected = linear(*context, input, weight, output);
    REQUIRE_FALSE(projected.has_value());
    CHECK(projected.error() == tensor_op_errc::output_shape_mismatch);
}

TEST_CASE("linear preserves signed fractional bf16 weights")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

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

TEST_CASE("embedding lookup gathers token rows in input order")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

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

TEST_CASE("embedding lookup requires rank-one ids and rank-two matrices")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    SUBCASE("token ids")
    {
        auto token_ids = make_tensor(*context, dtype::i32, { 1, 2 });
        auto weight = make_tensor(*context, dtype::bf16, { 3, 4 });
        auto output = make_tensor(*context, dtype::f32, { 2, 4 });

        auto gathered = embedding_lookup(*context, token_ids, weight, output);
        REQUIRE_FALSE(gathered.has_value());
        CHECK(gathered.error() == tensor_op_errc::invalid_rank);
    }

    SUBCASE("weight")
    {
        auto token_ids = make_tensor(*context, dtype::i32, { 2 });
        auto weight = make_tensor(*context, dtype::bf16, { 12 });
        auto output = make_tensor(*context, dtype::f32, { 2, 4 });

        auto gathered = embedding_lookup(*context, token_ids, weight, output);
        REQUIRE_FALSE(gathered.has_value());
        CHECK(gathered.error() == tensor_op_errc::invalid_rank);
    }

    SUBCASE("output")
    {
        auto token_ids = make_tensor(*context, dtype::i32, { 2 });
        auto weight = make_tensor(*context, dtype::bf16, { 3, 4 });
        auto output = make_tensor(*context, dtype::f32, { 8 });

        auto gathered = embedding_lookup(*context, token_ids, weight, output);
        REQUIRE_FALSE(gathered.has_value());
        CHECK(gathered.error() == tensor_op_errc::invalid_rank);
    }
}

TEST_CASE("embedding lookup requires i32 ids bf16 weights and f32 output")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    SUBCASE("token ids")
    {
        auto token_ids = make_tensor(*context, dtype::f32, { 2 });
        auto weight = make_tensor(*context, dtype::bf16, { 3, 4 });
        auto output = make_tensor(*context, dtype::f32, { 2, 4 });

        auto gathered = embedding_lookup(*context, token_ids, weight, output);
        REQUIRE_FALSE(gathered.has_value());
        CHECK(gathered.error() == tensor_op_errc::unsupported_dtype);
    }

    SUBCASE("weight")
    {
        auto token_ids = make_tensor(*context, dtype::i32, { 2 });
        auto weight = make_tensor(*context, dtype::f32, { 3, 4 });
        auto output = make_tensor(*context, dtype::f32, { 2, 4 });

        auto gathered = embedding_lookup(*context, token_ids, weight, output);
        REQUIRE_FALSE(gathered.has_value());
        CHECK(gathered.error() == tensor_op_errc::unsupported_dtype);
    }

    SUBCASE("output")
    {
        auto token_ids = make_tensor(*context, dtype::i32, { 2 });
        auto weight = make_tensor(*context, dtype::bf16, { 3, 4 });
        auto output = make_tensor(*context, dtype::bf16, { 2, 4 });

        auto gathered = embedding_lookup(*context, token_ids, weight, output);
        REQUIRE_FALSE(gathered.has_value());
        CHECK(gathered.error() == tensor_op_errc::unsupported_dtype);
    }
}

TEST_CASE("embedding lookup requires an exact output shape")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    SUBCASE("token count")
    {
        auto token_ids = make_tensor(*context, dtype::i32, { 2 });
        auto weight = make_tensor(*context, dtype::bf16, { 3, 4 });
        auto output = make_tensor(*context, dtype::f32, { 1, 4 });

        auto gathered = embedding_lookup(*context, token_ids, weight, output);
        REQUIRE_FALSE(gathered.has_value());
        CHECK(gathered.error() == tensor_op_errc::output_shape_mismatch);
    }

    SUBCASE("hidden size")
    {
        auto token_ids = make_tensor(*context, dtype::i32, { 2 });
        auto weight = make_tensor(*context, dtype::bf16, { 3, 4 });
        auto output = make_tensor(*context, dtype::f32, { 2, 3 });

        auto gathered = embedding_lookup(*context, token_ids, weight, output);
        REQUIRE_FALSE(gathered.has_value());
        CHECK(gathered.error() == tensor_op_errc::output_shape_mismatch);
    }
}

TEST_CASE("embedding lookup rejects negative token ids")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto token_ids = make_tensor(*context, dtype::i32, { 2 });
    auto weight = make_tensor(*context, dtype::bf16, { 3, 4 });
    auto output = make_tensor(*context, dtype::f32, { 2, 4 });
    write_i32(token_ids, { 0, -1 });

    auto gathered = embedding_lookup(*context, token_ids, weight, output);
    REQUIRE_FALSE(gathered.has_value());
    CHECK(gathered.error() == tensor_op_errc::token_out_of_range);
}

TEST_CASE("embedding lookup rejects token ids beyond the vocabulary")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto token_ids = make_tensor(*context, dtype::i32, { 2 });
    auto weight = make_tensor(*context, dtype::bf16, { 3, 4 });
    auto output = make_tensor(*context, dtype::f32, { 2, 4 });
    write_i32(token_ids, { 0, 3 });

    auto gathered = embedding_lookup(*context, token_ids, weight, output);
    REQUIRE_FALSE(gathered.has_value());
    CHECK(gathered.error() == tensor_op_errc::token_out_of_range);
}

TEST_CASE("rms norm normalizes and scales each row independently")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

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

TEST_CASE("rms norm requires rank-two activations and a rank-one weight")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    SUBCASE("input")
    {
        auto input = make_tensor(*context, dtype::f32, { 6 });
        auto weight = make_tensor(*context, dtype::bf16, { 3 });
        auto output = make_tensor(*context, dtype::f32, { 2, 3 });

        auto normalized = rms_norm(*context, input, weight, 1.0F, output);
        REQUIRE_FALSE(normalized.has_value());
        CHECK(normalized.error() == tensor_op_errc::invalid_rank);
    }

    SUBCASE("weight")
    {
        auto input = make_tensor(*context, dtype::f32, { 2, 3 });
        auto weight = make_tensor(*context, dtype::bf16, { 1, 3 });
        auto output = make_tensor(*context, dtype::f32, { 2, 3 });

        auto normalized = rms_norm(*context, input, weight, 1.0F, output);
        REQUIRE_FALSE(normalized.has_value());
        CHECK(normalized.error() == tensor_op_errc::invalid_rank);
    }

    SUBCASE("output")
    {
        auto input = make_tensor(*context, dtype::f32, { 2, 3 });
        auto weight = make_tensor(*context, dtype::bf16, { 3 });
        auto output = make_tensor(*context, dtype::f32, { 6 });

        auto normalized = rms_norm(*context, input, weight, 1.0F, output);
        REQUIRE_FALSE(normalized.has_value());
        CHECK(normalized.error() == tensor_op_errc::invalid_rank);
    }
}

TEST_CASE("rms norm requires f32 activations and a bf16 weight")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    SUBCASE("input")
    {
        auto input = make_tensor(*context, dtype::bf16, { 2, 3 });
        auto weight = make_tensor(*context, dtype::bf16, { 3 });
        auto output = make_tensor(*context, dtype::f32, { 2, 3 });

        auto normalized = rms_norm(*context, input, weight, 1.0F, output);
        REQUIRE_FALSE(normalized.has_value());
        CHECK(normalized.error() == tensor_op_errc::unsupported_dtype);
    }

    SUBCASE("weight")
    {
        auto input = make_tensor(*context, dtype::f32, { 2, 3 });
        auto weight = make_tensor(*context, dtype::f32, { 3 });
        auto output = make_tensor(*context, dtype::f32, { 2, 3 });

        auto normalized = rms_norm(*context, input, weight, 1.0F, output);
        REQUIRE_FALSE(normalized.has_value());
        CHECK(normalized.error() == tensor_op_errc::unsupported_dtype);
    }

    SUBCASE("output")
    {
        auto input = make_tensor(*context, dtype::f32, { 2, 3 });
        auto weight = make_tensor(*context, dtype::bf16, { 3 });
        auto output = make_tensor(*context, dtype::bf16, { 2, 3 });

        auto normalized = rms_norm(*context, input, weight, 1.0F, output);
        REQUIRE_FALSE(normalized.has_value());
        CHECK(normalized.error() == tensor_op_errc::unsupported_dtype);
    }
}

TEST_CASE("rms norm requires the weight to match the hidden size")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto input = make_tensor(*context, dtype::f32, { 2, 3 });
    auto weight = make_tensor(*context, dtype::bf16, { 4 });
    auto output = make_tensor(*context, dtype::f32, { 2, 3 });

    auto normalized = rms_norm(*context, input, weight, 1.0F, output);
    REQUIRE_FALSE(normalized.has_value());
    CHECK(normalized.error() == tensor_op_errc::inner_dimension_mismatch);
}

TEST_CASE("rms norm requires output to match the input shape")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    SUBCASE("row count")
    {
        auto input = make_tensor(*context, dtype::f32, { 2, 3 });
        auto weight = make_tensor(*context, dtype::bf16, { 3 });
        auto output = make_tensor(*context, dtype::f32, { 1, 3 });

        auto normalized = rms_norm(*context, input, weight, 1.0F, output);
        REQUIRE_FALSE(normalized.has_value());
        CHECK(normalized.error() == tensor_op_errc::output_shape_mismatch);
    }

    SUBCASE("hidden size")
    {
        auto input = make_tensor(*context, dtype::f32, { 2, 3 });
        auto weight = make_tensor(*context, dtype::bf16, { 3 });
        auto output = make_tensor(*context, dtype::f32, { 2, 4 });

        auto normalized = rms_norm(*context, input, weight, 1.0F, output);
        REQUIRE_FALSE(normalized.has_value());
        CHECK(normalized.error() == tensor_op_errc::output_shape_mismatch);
    }
}

TEST_CASE("rms norm requires a positive finite epsilon")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto input = make_tensor(*context, dtype::f32, { 2, 3 });
    auto weight = make_tensor(*context, dtype::bf16, { 3 });
    auto output = make_tensor(*context, dtype::f32, { 2, 3 });

    const std::vector<float> invalid_values {
        0.0F,
        -1.0F,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };

    for (const auto epsilon : invalid_values) {
        auto normalized = rms_norm(*context, input, weight, epsilon, output);
        REQUIRE_FALSE(normalized.has_value());
        CHECK(normalized.error() == tensor_op_errc::invalid_epsilon);
    }
}

TEST_CASE("silu multiply gates the up projection elementwise")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

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

TEST_CASE("silu multiply requires rank-two tensors")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    SUBCASE("gate")
    {
        auto gate = make_tensor(*context, dtype::f32, { 4 });
        auto up = make_tensor(*context, dtype::f32, { 2, 2 });
        auto output = make_tensor(*context, dtype::f32, { 2, 2 });

        auto activated = silu_mul(*context, gate, up, output);
        REQUIRE_FALSE(activated.has_value());
        CHECK(activated.error() == tensor_op_errc::invalid_rank);
    }

    SUBCASE("up")
    {
        auto gate = make_tensor(*context, dtype::f32, { 2, 2 });
        auto up = make_tensor(*context, dtype::f32, { 4 });
        auto output = make_tensor(*context, dtype::f32, { 2, 2 });

        auto activated = silu_mul(*context, gate, up, output);
        REQUIRE_FALSE(activated.has_value());
        CHECK(activated.error() == tensor_op_errc::invalid_rank);
    }

    SUBCASE("output")
    {
        auto gate = make_tensor(*context, dtype::f32, { 2, 2 });
        auto up = make_tensor(*context, dtype::f32, { 2, 2 });
        auto output = make_tensor(*context, dtype::f32, { 4 });

        auto activated = silu_mul(*context, gate, up, output);
        REQUIRE_FALSE(activated.has_value());
        CHECK(activated.error() == tensor_op_errc::invalid_rank);
    }
}

TEST_CASE("silu multiply requires f32 tensors")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    SUBCASE("gate")
    {
        auto gate = make_tensor(*context, dtype::bf16, { 2, 2 });
        auto up = make_tensor(*context, dtype::f32, { 2, 2 });
        auto output = make_tensor(*context, dtype::f32, { 2, 2 });

        auto activated = silu_mul(*context, gate, up, output);
        REQUIRE_FALSE(activated.has_value());
        CHECK(activated.error() == tensor_op_errc::unsupported_dtype);
    }

    SUBCASE("up")
    {
        auto gate = make_tensor(*context, dtype::f32, { 2, 2 });
        auto up = make_tensor(*context, dtype::bf16, { 2, 2 });
        auto output = make_tensor(*context, dtype::f32, { 2, 2 });

        auto activated = silu_mul(*context, gate, up, output);
        REQUIRE_FALSE(activated.has_value());
        CHECK(activated.error() == tensor_op_errc::unsupported_dtype);
    }

    SUBCASE("output")
    {
        auto gate = make_tensor(*context, dtype::f32, { 2, 2 });
        auto up = make_tensor(*context, dtype::f32, { 2, 2 });
        auto output = make_tensor(*context, dtype::bf16, { 2, 2 });

        auto activated = silu_mul(*context, gate, up, output);
        REQUIRE_FALSE(activated.has_value());
        CHECK(activated.error() == tensor_op_errc::unsupported_dtype);
    }
}

TEST_CASE("silu multiply requires matching input shapes")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto gate = make_tensor(*context, dtype::f32, { 2, 3 });
    auto up = make_tensor(*context, dtype::f32, { 3, 2 });
    auto output = make_tensor(*context, dtype::f32, { 2, 3 });

    auto activated = silu_mul(*context, gate, up, output);
    REQUIRE_FALSE(activated.has_value());
    CHECK(activated.error() == tensor_op_errc::input_shape_mismatch);
}

TEST_CASE("silu multiply requires output to match the inputs")
{
    auto context = metal_context::make(load_shader_source());
    REQUIRE(context.has_value());

    auto gate = make_tensor(*context, dtype::f32, { 2, 3 });
    auto up = make_tensor(*context, dtype::f32, { 2, 3 });
    auto output = make_tensor(*context, dtype::f32, { 3, 2 });

    auto activated = silu_mul(*context, gate, up, output);
    REQUIRE_FALSE(activated.has_value());
    CHECK(activated.error() == tensor_op_errc::output_shape_mismatch);
}
