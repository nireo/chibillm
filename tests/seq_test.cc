#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <expected>
#include <string>
#include <vector>

#include "seq.h"

TEST_CASE("sampling parameters have sensible defaults")
{
    const chibillm::sampling_params params;

    CHECK(params.temperature == 1.0f);
    CHECK(params.max_new_tokens == 64);
    CHECK_FALSE(params.ignore_eos);
}

TEST_CASE("a sequence can be constructed")
{
    const std::vector<chibillm::token_id> prompt { 1, 2, 3 };
    const chibillm::sampling_params params {};
    auto result = chibillm::seq::make(1, prompt, params, 16);

    REQUIRE(result.has_value());
    const auto& sequence = result.value();
    CHECK(sequence.id() == 1);
    CHECK(sequence.status() == chibillm::seq_status::waiting);
    CHECK(sequence.token_count() == 3);
    CHECK(sequence.completion_token_count() == 0);
    CHECK(sequence.block_size() == 16);
}

TEST_CASE("an empty prompt returns an expected error")
{
    auto result = chibillm::seq::make(
        1,
        std::vector<chibillm::token_id> {},
        chibillm::sampling_params {},
        16);

    REQUIRE_FALSE(result.has_value());
    CHECK(result.error() == chibillm::seq_errc::empty_prompt);
}

TEST_CASE("C++23 standard library support is available")
{
    const std::expected<int, std::string> result = 42;

    REQUIRE(result.has_value());
    CHECK(result.value() == 42);
}
