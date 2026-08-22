#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "metal/metal_context.h"
#include "metal/metal_tensor.h"
#include "tensor/dtype.h"
#include "tensor/tensor_descriptor.h"
#include "tensor/tensor_ops.h"
#include "tensor/tensor_shape.h"

using chibillm::dtype;
using chibillm::matmul;
using chibillm::metal_context;
using chibillm::metal_tensor;
using chibillm::tensor_descriptor;
using chibillm::tensor_op_errc;
using chibillm::tensor_shape;

namespace {

std::string
load_shader_source()
{
    std::ifstream input(CHIBILLM_VECTOR_ADD_SHADER_PATH);
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
