#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <fstream>
#include <iterator>
#include <string>

#include "metal/metal_context.h"

using chibillm::metal_context;

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
