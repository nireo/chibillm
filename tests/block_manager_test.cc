#include "block_manager.h"
#include <doctest/doctest.h>
#include <limits>
using namespace chibillm;

TEST_CASE("paged state validates geometry")
{
    CHECK(block_manager::make(0, 16).error() == block_manager_errc::invalid_block_count);
    CHECK(block_manager::make(4, 0).error() == block_manager_errc::invalid_block_size);
    CHECK(
        block_manager::make(static_cast<std::size_t>(std::numeric_limits<block_id>::max()) + 2, 16)
            .error()
        == block_manager_errc::too_many_blocks);
}

TEST_CASE("paged reservations grow at block boundaries and release without sharing ownership")
{
    auto state = block_manager::make(4, 2);
    REQUIRE(state.has_value());
    REQUIRE(state->reserve(1, 3).has_value());
    CHECK(state->resources(1).blocks.size() == 2);
    CHECK(state->free_block_count() == 2);
    REQUIRE(state->reserve(1, 4).has_value());
    CHECK(state->free_block_count() == 2);
    REQUIRE(state->reserve(1, 5).has_value());
    CHECK(state->resources(1).blocks.size() == 3);
    REQUIRE(state->reserve(2, 2).has_value());
    CHECK(state->resources(2).blocks[0] == 3);
    state->release(1);
    CHECK(state->resources(1).blocks.empty());
    CHECK(state->free_block_count() == 3);
    state->release(1);
    CHECK(state->free_block_count() == 3);
    REQUIRE(state->reserve(3, 1).has_value());
    CHECK(state->resources(3).blocks[0] == 0);
    CHECK(state->resources(2).blocks[0] == 3);
}

TEST_CASE("failed reservations leave existing allocations intact")
{
    auto state = block_manager::make(2, 2);
    REQUIRE(state.has_value());
    REQUIRE(state->reserve(1, 2).has_value());
    CHECK(state->reserve(2, 3).error() == state_errc::capacity_exhausted);
    CHECK(state->resources(2).blocks.empty());
    CHECK(state->reserve(1, 5).error() == state_errc::capacity_exhausted);
    CHECK(state->resources(1).blocks.size() == 1);
    CHECK(state->free_block_count() == 1);
    CHECK(state->reserve(2, 0).error() == state_errc::invalid_reservation);
}
