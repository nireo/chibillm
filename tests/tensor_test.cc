#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "metal/metal_tensor.h"
#include "tensor/bf16.h"
#include "tensor/dtype.h"
#include "tensor/tensor_descriptor.h"
#include "tensor/tensor_shape.h"

using chibillm::bf16;
using chibillm::dtype;
using chibillm::element_size;
using chibillm::metal_context;
using chibillm::metal_tensor;
using chibillm::tensor_descriptor;
using chibillm::tensor_errc;
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

} // namespace

TEST_CASE("dtype sizes are explicit and invalid values are rejected")
{
    auto f32_size = element_size(dtype::f32);
    auto bf16_size = element_size(dtype::bf16);
    auto i32_size = element_size(dtype::i32);
    auto u32_size = element_size(dtype::u32);
    REQUIRE(f32_size.has_value());
    REQUIRE(bf16_size.has_value());
    REQUIRE(i32_size.has_value());
    REQUIRE(u32_size.has_value());
    CHECK(*f32_size == 4);
    CHECK(*bf16_size == 2);
    CHECK(*i32_size == 4);
    CHECK(*u32_size == 4);

    auto invalid = element_size(static_cast<dtype>(255));
    REQUIRE_FALSE(invalid.has_value());
    CHECK(invalid.error() == tensor_errc::unsupported_dtype);
}

TEST_CASE("bf16 bit patterns expand exactly to float")
{
    CHECK(bf16::from_bits(0x0000).to_float() == 0.0F);
    CHECK(bf16::from_bits(0x3f80).to_float() == 1.0F);
    CHECK(bf16::from_bits(0xc020).to_float() == -2.5F);
    CHECK(bf16::from_bits(0x3f80).bits() == 0x3f80);
}

TEST_CASE("float conversion uses round-to-nearest-even and preserves special values")
{
    CHECK(bf16::from_float(1.0F).bits() == 0x3f80);
    CHECK(bf16::from_float(-2.5F).bits() == 0xc020);

    const auto halfway_to_even_low = std::bit_cast<float>(std::uint32_t { 0x3f808000 });
    const auto halfway_to_even_high = std::bit_cast<float>(std::uint32_t { 0x3f818000 });
    CHECK(bf16::from_float(halfway_to_even_low).bits() == 0x3f80);
    CHECK(bf16::from_float(halfway_to_even_high).bits() == 0x3f82);

    const auto infinity = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(std::isinf(bf16::from_float(infinity).to_float()));
    CHECK(std::isnan(bf16::from_float(nan).to_float()));
}

TEST_CASE("tensor shape owns dimensions and precomputes the element count")
{
    auto shape = tensor_shape::make({ 2, 3, 4 });
    REQUIRE(shape.has_value());
    CHECK(shape->rank() == 3);
    CHECK(shape->element_count() == 24);
    REQUIRE(shape->dimensions().size() == 3);
    CHECK(shape->dimensions()[0] == 2);
    CHECK(shape->dimensions()[1] == 3);
    CHECK(shape->dimensions()[2] == 4);

    auto second_dimension = shape->dimension(1);
    REQUIRE(second_dimension.has_value());
    CHECK(*second_dimension == 3);

    auto invalid_axis = shape->dimension(3);
    REQUIRE_FALSE(invalid_axis.has_value());
    CHECK(invalid_axis.error() == tensor_errc::axis_out_of_range);
}

TEST_CASE("tensor shape rejects empty, zero, and overflowing dimensions")
{
    auto empty = tensor_shape::make({});
    REQUIRE_FALSE(empty.has_value());
    CHECK(empty.error() == tensor_errc::empty_shape);

    auto zero = tensor_shape::make({ 2, 0, 4 });
    REQUIRE_FALSE(zero.has_value());
    CHECK(zero.error() == tensor_errc::zero_dimension);

    auto overflow =
        tensor_shape::make({ std::numeric_limits<std::size_t>::max(), std::size_t { 2 } });
    REQUIRE_FALSE(overflow.has_value());
    CHECK(overflow.error() == tensor_errc::element_count_overflow);
}

TEST_CASE("tensor descriptor combines dtype shape and checked byte size")
{
    auto shape = tensor_shape::make({ 3, 1024 });
    REQUIRE(shape.has_value());

    auto descriptor = tensor_descriptor::make(dtype::bf16, std::move(*shape));
    REQUIRE(descriptor.has_value());
    CHECK(descriptor->type() == dtype::bf16);
    CHECK(descriptor->shape().rank() == 2);
    CHECK(descriptor->element_count() == 3072);
    CHECK(descriptor->size_bytes() == 6144);
}

TEST_CASE("tensor descriptor rejects byte overflow and invalid dtype")
{
    auto overflow_shape = tensor_shape::make({ std::numeric_limits<std::size_t>::max() });
    REQUIRE(overflow_shape.has_value());
    auto overflow = tensor_descriptor::make(dtype::f32, std::move(*overflow_shape));
    REQUIRE_FALSE(overflow.has_value());
    CHECK(overflow.error() == tensor_errc::byte_size_overflow);

    auto invalid_shape = tensor_shape::make({ 1 });
    REQUIRE(invalid_shape.has_value());
    auto invalid = tensor_descriptor::make(static_cast<dtype>(255), std::move(*invalid_shape));
    REQUIRE_FALSE(invalid.has_value());
    CHECK(invalid.error() == tensor_errc::unsupported_dtype);
}

TEST_CASE("metal tensor allocates exactly the descriptor byte size")
{
    static_assert(!std::is_copy_constructible_v<metal_tensor>);
    static_assert(std::is_move_constructible_v<metal_tensor>);

    const auto shader_source = load_shader_source();
    auto context = metal_context::make(shader_source);
    REQUIRE(context.has_value());
    CHECK_FALSE(context->device_name().empty());

    auto shape = tensor_shape::make({ 2, 3 });
    REQUIRE(shape.has_value());
    auto descriptor = tensor_descriptor::make(dtype::bf16, std::move(*shape));
    REQUIRE(descriptor.has_value());

    auto tensor = metal_tensor::make(*context, std::move(*descriptor));
    REQUIRE(tensor.has_value());
    CHECK(tensor->descriptor().type() == dtype::bf16);
    CHECK(tensor->descriptor().element_count() == 6);
    CHECK(tensor->descriptor().size_bytes() == 12);
    CHECK(tensor->buffer().size_bytes() == 12);
    CHECK(tensor->buffer().bytes().size() == 12);
}
