#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "metal/metal_context.h"

using chibillm::metal_context;
using chibillm::metal_errc;

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

void
write_floats(chibillm::metal_buffer& buffer, const std::vector<float>& values)
{
    REQUIRE(buffer.size_bytes() >= values.size() * sizeof(float));
    std::memcpy(buffer.bytes().data(), values.data(), values.size() * sizeof(float));
}

std::vector<float>
read_floats(const chibillm::metal_buffer& buffer, std::size_t count)
{
    REQUIRE(buffer.size_bytes() >= count * sizeof(float));

    std::vector<float> values(count);
    std::memcpy(values.data(), buffer.bytes().data(), count * sizeof(float));
    return values;
}

} // namespace

TEST_CASE("Metal context exposes the selected GPU")
{
    const auto shader_source = load_shader_source();
    REQUIRE_FALSE(shader_source.empty());

    auto context = metal_context::make(shader_source);
    const auto context_message = context.has_value() ? std::string {} : context.error().message;
    INFO(context_message);
    REQUIRE(context.has_value());
    CHECK_FALSE(context->device_name().empty());
}

TEST_CASE("Metal vector addition handles a partial final threadgroup")
{
    const auto shader_source = load_shader_source();
    auto context = metal_context::make(shader_source);
    REQUIRE(context.has_value());

    constexpr std::size_t element_count = 1000;
    const auto size_bytes = element_count * sizeof(float);

    std::vector<float> lhs(element_count);
    std::vector<float> rhs(element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        lhs[index] = static_cast<float>(index);
        rhs[index] = static_cast<float>(index) * 0.5F;
    }

    auto lhs_buffer = context->make_shared_buffer(size_bytes);
    auto rhs_buffer = context->make_shared_buffer(size_bytes);
    auto output_buffer = context->make_shared_buffer(size_bytes);
    REQUIRE(lhs_buffer.has_value());
    REQUIRE(rhs_buffer.has_value());
    REQUIRE(output_buffer.has_value());

    write_floats(*lhs_buffer, lhs);
    write_floats(*rhs_buffer, rhs);
    std::fill(output_buffer->bytes().begin(), output_buffer->bytes().end(), std::byte { 0 });

    auto dispatched =
        context->dispatch_vector_add(*lhs_buffer, *rhs_buffer, *output_buffer, element_count);
    const auto dispatch_message =
        dispatched.has_value() ? std::string {} : dispatched.error().message;
    INFO(dispatch_message);
    REQUIRE(dispatched.has_value());

    const auto output = read_floats(*output_buffer, element_count);
    for (std::size_t index = 0; index < element_count; ++index) {
        CHECK(output[index] == doctest::Approx(lhs[index] + rhs[index]));
    }
}

TEST_CASE("Metal vector addition rejects undersized buffers")
{
    const auto shader_source = load_shader_source();
    auto context = metal_context::make(shader_source);
    REQUIRE(context.has_value());

    auto lhs = context->make_shared_buffer(sizeof(float));
    auto rhs = context->make_shared_buffer(sizeof(float));
    auto output = context->make_shared_buffer(sizeof(float));
    REQUIRE(lhs.has_value());
    REQUIRE(rhs.has_value());
    REQUIRE(output.has_value());

    auto dispatched = context->dispatch_vector_add(*lhs, *rhs, *output, 2);
    REQUIRE_FALSE(dispatched.has_value());
    CHECK(dispatched.error().code == metal_errc::invalid_input);
}
