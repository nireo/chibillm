#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "metal_test_support.h"

#include <limits>

#include "metal/metal_kv_cache.h"
#include "tensor/dtype.h"

using chibillm::dtype;
using chibillm::kv_cache_config;
using chibillm::kv_cache_errc;
using chibillm::metal_context;
using chibillm::metal_kv_cache;

namespace {

kv_cache_config
valid_config()
{
    return {
        .layer_count = 2,
        .block_count = 3,
        .block_size = 4,
        .kv_head_count = 2,
        .head_dimension = 8,
    };
}

} // namespace

TEST_CASE("metal kv cache allocates matching key and value tensors")
{
    const auto& context = metal_test::test_context();
    auto cache = metal_kv_cache::make(context, valid_config());
    REQUIRE(cache.has_value());

    CHECK(cache->layer_count() == 2);
    CHECK(cache->block_count() == 3);
    CHECK(cache->block_size() == 4);
    CHECK(cache->kv_head_count() == 2);
    CHECK(cache->head_dimension() == 8);
    CHECK(cache->elements_per_token() == 16);
    CHECK(cache->elements_per_block() == 64);
    CHECK(cache->elements_per_layer() == 192);
    CHECK(cache->element_count() == 384);

    CHECK(cache->keys().descriptor().type() == dtype::f32);
    CHECK(cache->values().descriptor().type() == dtype::f32);
    CHECK(cache->keys().descriptor().size_bytes() == 384 * sizeof(float));
    CHECK(cache->values().descriptor().size_bytes() == 384 * sizeof(float));

    const auto dimensions = cache->keys().descriptor().shape().dimensions();
    REQUIRE(dimensions.size() == 5);
    CHECK(dimensions[0] == 2);
    CHECK(dimensions[1] == 3);
    CHECK(dimensions[2] == 4);
    CHECK(dimensions[3] == 2);
    CHECK(dimensions[4] == 8);
}

TEST_CASE("metal kv cache rejects zero and overflowing dimensions")
{
    const auto& context = metal_test::test_context();
    auto check_invalid = [&](auto mutator, kv_cache_errc expected) {
        auto cfg = valid_config();
        mutator(cfg);
        CHECK(metal_kv_cache::make(context, cfg).error() == expected);
    };

    check_invalid([](auto& c) { c.layer_count = 0; }, kv_cache_errc::invalid_layer_count);
    check_invalid([](auto& c) { c.block_count = 0; }, kv_cache_errc::invalid_block_count);
    check_invalid([](auto& c) { c.block_size = 0; }, kv_cache_errc::invalid_block_size);
    check_invalid([](auto& c) { c.kv_head_count = 0; }, kv_cache_errc::invalid_kv_head_count);
    check_invalid([](auto& c) { c.head_dimension = 0; }, kv_cache_errc::invalid_head_dimension);
    check_invalid([](auto& c) { c.layer_count = std::numeric_limits<std::size_t>::max(); },
                  kv_cache_errc::layout_size_overflow);
}

TEST_CASE("metal kv cache flattens coordinates in layout order and rejects invalid coordinates")
{
    const auto& context = metal_test::test_context();
    auto cache = metal_kv_cache::make(context, valid_config());
    REQUIRE(cache.has_value());

    CHECK(cache->element_offset(0, 0, 0, 0, 0) == 0);
    CHECK(cache->element_offset(0, 0, 0, 0, 1) == 1);
    CHECK(cache->element_offset(0, 0, 0, 1, 0) == 8);
    CHECK(cache->element_offset(0, 0, 1, 0, 0) == 16);
    CHECK(cache->element_offset(0, 1, 0, 0, 0) == 64);
    CHECK(cache->element_offset(1, 0, 0, 0, 0) == 192);
    CHECK(cache->element_offset(1, 2, 3, 1, 7) == 383);

    CHECK(cache->element_offset(2, 0, 0, 0, 0).error() == kv_cache_errc::layer_out_of_range);
    CHECK(cache->element_offset(0, 3, 0, 0, 0).error() == kv_cache_errc::block_out_of_range);
    CHECK(cache->element_offset(0, 0, 4, 0, 0).error() == kv_cache_errc::token_offset_out_of_range);
    CHECK(cache->element_offset(0, 0, 0, 2, 0).error() == kv_cache_errc::kv_head_out_of_range);
    CHECK(cache->element_offset(0, 0, 0, 0, 8).error() == kv_cache_errc::head_feature_out_of_range);
}

