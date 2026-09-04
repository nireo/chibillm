#pragma once

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "metal/metal_context.h"
#include "metal/metal_tensor.h"
#include "tensor/bf16.h"
#include "tensor/dtype.h"

namespace metal_test {

inline std::string
load_shader_source()
{
    std::ifstream input(CHIBILLM_SHADER_PATH);
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>(),
    };
}

inline const chibillm::metal_context&
test_context()
{
    static const auto context = [] {
        auto result = chibillm::metal_context::make(load_shader_source());
        if (!result) {
            throw std::runtime_error("failed to create metal context: " + result.error().message);
        }
        return std::move(*result);
    }();
    return context;
}

inline chibillm::metal_tensor
make_tensor(const chibillm::metal_context& context,
            chibillm::dtype type,
            std::vector<std::size_t> dimensions)
{
    auto tensor = chibillm::metal_tensor::make(context, type, std::move(dimensions));
    REQUIRE(tensor.has_value());
    return std::move(*tensor);
}

inline void
write_floats(chibillm::metal_tensor& tensor, const std::vector<float>& values)
{
    REQUIRE(tensor.descriptor().type() == chibillm::dtype::f32);
    REQUIRE(tensor.buffer().size_bytes() == values.size() * sizeof(float));
    std::memcpy(tensor.buffer().bytes().data(), values.data(), tensor.buffer().size_bytes());
}

inline void
write_f32(chibillm::metal_tensor& tensor, const std::vector<float>& values)
{
    write_floats(tensor, values);
}

inline void
write_bf16(chibillm::metal_tensor& tensor, const std::vector<float>& values)
{
    REQUIRE(tensor.descriptor().type() == chibillm::dtype::bf16);
    REQUIRE(tensor.buffer().size_bytes() == values.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> bits;
    bits.reserve(values.size());
    for (const auto value : values) {
        bits.push_back(chibillm::bf16::from_float(value).bits());
    }

    std::memcpy(tensor.buffer().bytes().data(), bits.data(), tensor.buffer().size_bytes());
}

inline void
write_i32(chibillm::metal_tensor& tensor, const std::vector<std::int32_t>& values)
{
    REQUIRE(tensor.descriptor().type() == chibillm::dtype::i32);
    REQUIRE(tensor.buffer().size_bytes() == values.size() * sizeof(std::int32_t));
    std::memcpy(tensor.buffer().bytes().data(), values.data(), tensor.buffer().size_bytes());
}

inline void
write_u32(chibillm::metal_tensor& tensor, const std::vector<std::uint32_t>& values)
{
    REQUIRE(tensor.descriptor().type() == chibillm::dtype::u32);
    REQUIRE(tensor.buffer().size_bytes() == values.size() * sizeof(std::uint32_t));
    std::memcpy(tensor.buffer().bytes().data(), values.data(), tensor.buffer().size_bytes());
}

inline std::vector<float>
read_floats(const chibillm::metal_tensor& tensor)
{
    REQUIRE(tensor.descriptor().type() == chibillm::dtype::f32);

    std::vector<float> values(tensor.descriptor().element_count());
    REQUIRE(tensor.buffer().size_bytes() == values.size() * sizeof(float));
    std::memcpy(values.data(), tensor.buffer().bytes().data(), tensor.buffer().size_bytes());
    return values;
}

inline void
check_floats(const chibillm::metal_tensor& tensor, const std::vector<float>& expected)
{
    const auto values = read_floats(tensor);
    REQUIRE(values.size() == expected.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        CHECK(values[index] == doctest::Approx(expected[index]));
    }
}

} // namespace metal_test
