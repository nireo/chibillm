#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "metal_test_support.h"
#include "tensor/functions.h"

using namespace chibillm;
using namespace metal_test;

namespace {

// Independent storage decoder: scalar indexing and double-precision reference
// products, without calling production quantization or GPU decoding helpers.
std::vector<float>
decode(const quantized_matrix& matrix)
{
    const auto packed = matrix.packed().buffer().bytes();
    const auto scales = matrix.scales().buffer().bytes();
    const auto offsets = matrix.offsets().buffer().bytes();
    const auto per_word = matrix.values_per_word();
    const auto bits = matrix.bits();
    std::vector<float> values(matrix.shape().element_count());
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint32_t word;
        std::uint16_t scale, offset;
        std::memcpy(&word, packed.data() + (i / per_word) * sizeof(word), sizeof(word));
        std::memcpy(&scale, scales.data() + (i / 64) * sizeof(scale), sizeof(scale));
        std::memcpy(&offset, offsets.data() + (i / 64) * sizeof(offset), sizeof(offset));
        values[i] = bf16::from_bits(scale).to_float()
                * ((word >> (bits * (i % per_word))) & ((1u << bits) - 1u))
            + bf16::from_bits(offset).to_float();
    }
    return values;
}

quantized_matrix
make_matrix(const metal_context& context,
            std::size_t outputs,
            std::size_t inputs,
            int seed = 0,
            std::uint32_t bits = 4)
{
    if (bits == 8) {
        auto packed = make_tensor(context, dtype::u32, { outputs, inputs / 4 });
        auto scales = make_tensor(context, dtype::bf16, { outputs, inputs / 64 });
        auto offsets = make_tensor(context, dtype::bf16, { outputs, inputs / 64 });
        std::vector<std::uint32_t> words(outputs * inputs / 4);
        std::vector<float> scale_values(outputs * inputs / 64), offset_values(scale_values.size());
        for (std::size_t o = 0; o < outputs; ++o) {
            for (std::size_t word = 0; word < inputs / 4; ++word)
                for (std::size_t i = 0; i < 4; ++i)
                    words[o * inputs / 4 + word] |=
                        std::uint32_t((o * 7 + (word * 4 + i) * 3 + seed) % 256) << (i * 8);
            for (std::size_t group = 0; group < inputs / 64; ++group) {
                const auto index = o * inputs / 64 + group;
                scale_values[index] = (o + group) % 2 == 0 ? 1.0F / 128 : -1.0F / 128;
                offset_values[index] = scale_values[index] > 0 ? -1.0F : 1.0F;
            }
        }
        write_u32(packed, words);
        write_bf16(scales, scale_values);
        write_bf16(offsets, offset_values);
        auto shape = tensor_shape::make({ outputs, inputs });
        REQUIRE(shape);
        auto matrix = quantized_matrix::from_packed(std::move(*shape), std::move(packed),
                                                    std::move(scales), std::move(offsets), 8);
        REQUIRE(matrix);
        return std::move(*matrix);
    }
    auto dense = make_tensor(context, dtype::bf16, { outputs, inputs });
    std::vector<float> values(outputs * inputs);
    for (std::size_t o = 0; o < outputs; ++o)
        for (std::size_t i = 0; i < inputs; ++i)
            values[o * inputs + i] =
                float(int((o * 7 + i * 3 + seed) % 16) - 8) * 0.125F + float(i / 64) * 0.25F;
    write_bf16(dense, values);
    auto matrix = quantized_matrix::quantize(context, dense);
    REQUIRE(matrix);
    return std::move(*matrix);
}

std::vector<float>
reference(const std::vector<float>& input, const quantized_matrix& matrix)
{
    const auto n = matrix.shape().dimensions()[0], k = matrix.shape().dimensions()[1];
    const auto weight = decode(matrix);
    std::vector<float> result(input.size() / k * n);
    for (std::size_t r = 0; r < input.size() / k; ++r)
        for (std::size_t o = 0; o < n; ++o) {
            double sum = 0;
            for (std::size_t i = 0; i < k; ++i)
                sum += double(input[r * k + i]) * weight[o * k + i];
            result[r * n + o] = static_cast<float>(sum);
        }
    return result;
}

void
near(const std::vector<float>& actual, const std::vector<float>& expected)
{
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CAPTURE(i);
        CHECK(actual[i] == doctest::Approx(expected[i]).epsilon(1e-5).scale(1.0));
    }
}

} // namespace

TEST_CASE("Q4 packs low bits first, accounts for metadata, and preserves constant groups")
{
    const auto& context = test_context();
    auto source = make_tensor(context, dtype::bf16, { 2, 192 });
    std::vector<float> values(384);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = i / 64 == 1 ? 0.0F : i / 64 == 2 ? -2.0F : float(i % 16) * 0.125F - 1.0F;
    write_bf16(source, values);
    auto quantized = quantized_matrix::quantize(context, source);
    REQUIRE(quantized);
    std::uint32_t first, second;
    std::memcpy(&first, quantized->packed().buffer().bytes().data(), sizeof(first));
    std::memcpy(&second, quantized->packed().buffer().bytes().data() + sizeof(first),
                sizeof(second));
    CHECK(first == 0x76543210u);
    CHECK(second == 0xfedcba98u);
    CHECK(quantized->size_bytes() == 384 * 9 / 16);
    CHECK(decode(*quantized) == values);
    matrix_weight owned(std::move(*quantized));
    CHECK(owned.shape().dimensions()[0] == 2);
    CHECK(owned.shape().dimensions()[1] == 192);
    CHECK_FALSE(owned.dense());
    CHECK(owned.size_bytes() == 216);
}

TEST_CASE("Q4 rejects invalid layouts and nonfinite source values")
{
    const auto& context = test_context();
    for (auto type : { dtype::f32, dtype::bf16 }) {
        auto source = make_tensor(context, type, { 2, 63 });
        const auto result = quantized_matrix::quantize(context, source);
        REQUIRE_FALSE(result);
        CHECK(result.error() == matrix_errc::invalid_layout);
    }
    auto source = make_tensor(context, dtype::bf16, { 1, 64 });
    auto wrong_type = make_tensor(context, dtype::f32, { 1, 64 });
    auto wrong_rank = make_tensor(context, dtype::bf16, { 64 });
    CHECK_FALSE(quantized_matrix::quantize(context, wrong_type));
    CHECK_FALSE(quantized_matrix::quantize(context, wrong_rank));
    for (const auto bad :
         { std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() }) {
        std::vector<float> values(64, 1.0F);
        values[63] = bad;
        write_bf16(source, values);
        const auto result = quantized_matrix::quantize(context, source);
        REQUIRE_FALSE(result);
        CHECK(result.error() == matrix_errc::nonfinite_weight);
    }
}

TEST_CASE("Q4 imports existing encodings and validates their storage")
{
    const auto& context = test_context();
    const auto import = [&](std::vector<std::size_t> dimensions, dtype packed_type,
                            std::vector<std::size_t> packed_shape, float scale) {
        auto shape = tensor_shape::make(std::move(dimensions));
        REQUIRE(shape);
        auto packed = make_tensor(context, packed_type, std::move(packed_shape));
        auto scales = make_tensor(context, dtype::bf16, { 2, 1 });
        auto offsets = make_tensor(context, dtype::bf16, { 2, 1 });
        if (packed_type == dtype::u32)
            write_u32(packed,
                      std::vector<std::uint32_t>(packed.descriptor().element_count(), 0x76543210u));
        write_bf16(scales, { scale, -0.125F });
        write_bf16(offsets, { -1.0F, 0.5F });
        return quantized_matrix::from_packed(std::move(*shape), std::move(packed),
                                             std::move(scales), std::move(offsets));
    };
    auto imported = import({ 2, 64 }, dtype::u32, { 2, 8 }, 0.25F);
    REQUIRE(imported);
    const auto values = decode(*imported);
    for (std::size_t i = 0; i < 64; ++i) {
        CHECK(values[i] == float(i % 8) * 0.25F - 1.0F);
        CHECK(values[64 + i] == float(i % 8) * -0.125F + 0.5F);
    }
    for (auto dimensions : { std::vector<std::size_t> { 128 }, { 2, 63 } }) {
        const auto result = import(std::move(dimensions), dtype::u32, { 2, 8 }, 0.25F);
        REQUIRE_FALSE(result);
        CHECK(result.error() == matrix_errc::invalid_layout);
    }
    CHECK_FALSE(import({ 2, 64 }, dtype::bf16, { 2, 8 }, 0.25F));
    CHECK_FALSE(import({ 2, 64 }, dtype::u32, { 2, 16 }, 0.25F));
    const auto nonfinite =
        import({ 2, 64 }, dtype::u32, { 2, 8 }, std::numeric_limits<float>::infinity());
    REQUIRE_FALSE(nonfinite);
    CHECK(nonfinite.error() == matrix_errc::nonfinite_weight);
}

TEST_CASE("Q4 minmax encoding bounds error using the stored scale")
{
    const auto& context = test_context();
    auto source = make_tensor(context, dtype::bf16, { 3, 128 });
    std::vector<float> values(384);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = bf16::from_float(float(std::sin(double(i) * 0.27) * 0.31)).to_float();
    write_bf16(source, values);
    auto matrix = quantized_matrix::quantize(context, source);
    REQUIRE(matrix);
    const auto reconstructed = decode(*matrix);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint16_t bits;
        std::memcpy(&bits, matrix->scales().buffer().bytes().data() + (i / 64) * sizeof(bits),
                    sizeof(bits));
        const auto step = bf16::from_bits(bits).to_float();
        CHECK(std::abs(values[i] - reconstructed[i]) <= step * 0.5F + 1e-7F);
    }
}

TEST_CASE("Q8 imports unsigned bytes low bits first and rejects invalid bit widths")
{
    const auto& context = test_context();
    const auto import = [&](std::uint32_t bits, std::size_t words) {
        auto shape = tensor_shape::make({ 2, 64 });
        REQUIRE(shape);
        auto packed = make_tensor(context, dtype::u32, { 2, words });
        auto scales = make_tensor(context, dtype::bf16, { 2, 1 });
        auto offsets = make_tensor(context, dtype::bf16, { 2, 1 });
        write_u32(packed, std::vector<std::uint32_t>(2 * words, 0xff807f00u));
        write_bf16(scales, { 1.0F / 128, -1.0F / 128 });
        write_bf16(offsets, { -1.0F, 1.0F });
        return quantized_matrix::from_packed(std::move(*shape), std::move(packed),
                                             std::move(scales), std::move(offsets), bits);
    };
    auto matrix = import(8, 16);
    REQUIRE(matrix);
    CHECK(matrix->bits() == 8);
    CHECK(matrix->size_bytes() == 136);
    const auto values = decode(*matrix);
    const std::array<float, 4> codes { 0, 127, 128, 255 };
    for (std::size_t i = 0; i < 64; ++i) {
        CHECK(values[i] == codes[i % 4] / 128 - 1);
        CHECK(values[64 + i] == 1 - codes[i % 4] / 128);
    }
    CHECK_FALSE(import(8, 8));
    for (const auto bits : { 0, 2, 16 })
        CHECK_FALSE(import(bits, 16));
}

TEST_CASE("Q4/Q8 projections preserve split layout and residuals across decode and prefill")
{
    std::uint32_t bits = 4;
    SUBCASE("Q4") {}
    SUBCASE("Q8")
    {
        bits = 8;
    }
    auto made = metal_context::make(kernel_source());
    REQUIRE(made);
    auto& context = *made;
    auto weight = make_matrix(context, 73, 192, 0, bits);
    auto other = make_matrix(context, 73, 192, 5, bits);
    for (const std::size_t rows : { 1, 3, 8, 65 }) {
        CAPTURE(rows);
        std::vector<float> values(rows * 192);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = float(std::sin(double(i) * 0.1));
        auto input = make_tensor(context, dtype::f32, { rows, 192 });
        auto residual = make_tensor(context, dtype::f32, { rows, 73 });
        write_floats(input, values);
        write_floats(residual, std::vector<float>(rows * 73, 0.5F));
        compute_pass pass(context);
        REQUIRE(pass.begin());
        auto split = linear_split(context, input, weight, 17, 33, 23);
        REQUIRE(split);
        auto added = linear_add(context, input, other, residual);
        REQUIRE(added);
        // Another expansion must not overwrite weights consumed by earlier work.
        auto repeated = linear(context, input, weight);
        REQUIRE(repeated);
        REQUIRE(pass.finish());
        const auto expected = reference(values, weight);
        const auto a = read_floats((*split)[0]), b = read_floats((*split)[1]),
                   c = read_floats((*split)[2]);
        std::vector<float> joined;
        for (std::size_t r = 0; r < rows; ++r) {
            joined.insert(joined.end(), a.begin() + r * 17, a.begin() + (r + 1) * 17);
            joined.insert(joined.end(), b.begin() + r * 33, b.begin() + (r + 1) * 33);
            joined.insert(joined.end(), c.begin() + r * 23, c.begin() + (r + 1) * 23);
        }
        near(joined, expected);
        near(read_floats(*repeated), expected);
        auto expected_added = reference(values, other);
        for (auto& value : expected_added)
            value += 0.5F;
        near(read_floats(*added), expected_added);
    }
}

TEST_CASE("Q4/Q8 embeddings gather only requested rows and reject invalid tokens")
{
    std::uint32_t bits = 4;
    SUBCASE("Q4") {}
    SUBCASE("Q8")
    {
        bits = 8;
    }
    const auto& context = test_context();
    auto weight = make_matrix(context, 73, 64, 0, bits);
    const std::array<token_id, 4> tokens { 72, 3, 0, 3 };
    auto embedded = embed_tokens(context, weight, tokens);
    REQUIRE(embedded);
    const auto all = decode(weight);
    std::vector<float> expected;
    for (auto token : tokens)
        expected.insert(expected.end(), all.begin() + token * 64, all.begin() + (token + 1) * 64);
    CHECK(read_floats(*embedded) == expected);
    const std::array<token_id, 1> invalid { 73 };
    const auto bad = embed_tokens(context, weight, invalid);
    REQUIRE_FALSE(bad);
    CHECK(bad.error() == tensor_op_errc::token_out_of_range);
}

TEST_CASE("Q4/Q8 vocabulary argmax matches scalar scores and chooses the first tied token")
{
    std::uint32_t bits = 4;
    SUBCASE("Q4") {}
    SUBCASE("Q8")
    {
        bits = 8;
    }
    const auto& context = test_context();
    auto weight = make_matrix(context, 73, 64, 0, bits);
    auto norm = make_tensor(context, dtype::bf16, { 64 });
    write_bf16(norm, std::vector<float>(64, 1.0F));
    auto hidden = make_tensor(context, dtype::f32, { 3, 64 });
    std::vector<float> values(192);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = float(std::cos(double(i) * 0.2));
    write_floats(hidden, values);
    const std::array<std::size_t, 2> selected { 2, 0 };
    auto tokens = encode_greedy(context, norm, weight, 1e-6F, hidden, selected);
    REQUIRE(tokens);
    auto normalized = rms_norm(context, hidden, norm, 1e-6F);
    REQUIRE(normalized);
    const auto scores = reference(read_floats(*normalized), weight);
    std::vector<token_id> expected;
    for (auto row : selected) {
        auto begin = scores.begin() + row * 73;
        expected.push_back(static_cast<token_id>(std::max_element(begin, begin + 73) - begin));
    }
    CHECK(read_greedy(*tokens) == expected);
    write_floats(hidden, std::vector<float>(192, 0.0F));
    tokens = encode_greedy(context, norm, weight, 1e-6F, hidden, selected);
    REQUIRE(tokens);
    CHECK(read_greedy(*tokens) == std::vector<token_id> { 0, 0 });
}
