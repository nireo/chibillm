#include <doctest/doctest.h>

#include <limits>

#include "block_manager.h"

using chibillm::block_id;
using chibillm::block_manager;
using chibillm::block_manager_errc;
using chibillm::kv_block;
using chibillm::sampling_params;
using chibillm::seq;
using chibillm::seq_status;

TEST_CASE("KV block free state follows its reference count")
{
    kv_block block {
        .id = 3,
        .ref_count = 0,
    };

    CHECK(block.is_free());

    block.ref_count = 1;
    CHECK_FALSE(block.is_free());
}

TEST_CASE("block manager construction validates its geometry")
{
    auto zero_blocks = block_manager::make(0, 16);
    REQUIRE_FALSE(zero_blocks.has_value());
    CHECK(zero_blocks.error() == block_manager_errc::invalid_block_count);

    auto zero_size = block_manager::make(4, 0);
    REQUIRE_FALSE(zero_size.has_value());
    CHECK(zero_size.error() == block_manager_errc::invalid_block_size);

    if constexpr (
        std::numeric_limits<std::size_t>::max() >
        std::numeric_limits<block_id>::max()) {
        const auto unrepresentable =
            static_cast<std::size_t>(std::numeric_limits<block_id>::max()) + 2;
        auto too_many = block_manager::make(unrepresentable, 16);
        REQUIRE_FALSE(too_many.has_value());
        CHECK(too_many.error() == block_manager_errc::too_many_blocks);
    }
}

TEST_CASE("new block manager starts with every block free")
{
    auto result = block_manager::make(4, 16);
    REQUIRE(result.has_value());
    const auto& manager = *result;

    CHECK(manager.block_count() == 4);
    CHECK(manager.block_size() == 16);
    CHECK(manager.free_block_count() == 4);
    CHECK(manager.used_block_count() == 0);
    REQUIRE(manager.blocks().size() == 4);

    for (std::size_t index = 0; index < manager.blocks().size(); ++index) {
        CHECK(manager.blocks()[index].id == static_cast<block_id>(index));
        CHECK(manager.blocks()[index].ref_count == 0);
        CHECK(manager.blocks()[index].is_free());
    }
}

TEST_CASE("block requirements and capacity are calculated without mutation")
{
    auto manager_result = block_manager::make(4, 2);
    auto sequence_result = seq::make(1, { 1, 2, 3 }, sampling_params {}, 2);
    REQUIRE(manager_result.has_value());
    REQUIRE(sequence_result.has_value());
    auto& manager = *manager_result;
    const auto& sequence = *sequence_result;

    auto required = manager.additional_blocks_required(sequence);
    auto can_allocate = manager.can_ensure_capacity(sequence);

    REQUIRE(required.has_value());
    REQUIRE(can_allocate.has_value());
    CHECK(*required == 2);
    CHECK(*can_allocate);
    CHECK(manager.free_block_count() == 4);
    CHECK(sequence.block_table().empty());
}

TEST_CASE("ensure capacity assigns every missing physical block")
{
    auto manager_result = block_manager::make(4, 2);
    auto sequence_result = seq::make(1, { 1, 2, 3 }, sampling_params {}, 2);
    REQUIRE(manager_result.has_value());
    REQUIRE(sequence_result.has_value());
    auto& manager = *manager_result;
    auto& sequence = *sequence_result;

    REQUIRE(manager.ensure_capacity(sequence).has_value());
    REQUIRE(sequence.block_table().size() == 2);
    CHECK(sequence.block_table()[0] == 0);
    CHECK(sequence.block_table()[1] == 1);
    CHECK(manager.free_block_count() == 2);
    CHECK(manager.used_block_count() == 2);
    CHECK(manager.blocks()[0].ref_count == 1);
    CHECK(manager.blocks()[1].ref_count == 1);

    // Calling again when every logical block is already mapped is a no-op.
    REQUIRE(manager.ensure_capacity(sequence).has_value());
    CHECK(sequence.block_table().size() == 2);
    CHECK(manager.free_block_count() == 2);
}

TEST_CASE("insufficient capacity does not partially allocate")
{
    auto manager_result = block_manager::make(1, 2);
    auto sequence_result = seq::make(1, { 1, 2, 3 }, sampling_params {}, 2);
    REQUIRE(manager_result.has_value());
    REQUIRE(sequence_result.has_value());
    auto& manager = *manager_result;
    auto& sequence = *sequence_result;

    auto allocated = manager.ensure_capacity(sequence);

    REQUIRE_FALSE(allocated.has_value());
    CHECK(allocated.error() == block_manager_errc::insufficient_free_blocks);
    CHECK(sequence.block_table().empty());
    CHECK(manager.free_block_count() == 1);
    CHECK(manager.used_block_count() == 0);
    CHECK(manager.blocks()[0].is_free());
}

TEST_CASE("decode growth allocates only when crossing a block boundary")
{
    auto manager_result = block_manager::make(3, 2);
    auto sequence_result = seq::make(1, { 1, 2, 3 }, sampling_params {}, 2);
    REQUIRE(manager_result.has_value());
    REQUIRE(sequence_result.has_value());
    auto& manager = *manager_result;
    auto& sequence = *sequence_result;

    REQUIRE(manager.ensure_capacity(sequence).has_value());
    REQUIRE(sequence.schedule_tokens(3).has_value());
    REQUIRE(sequence.commit_scheduled_tokens().has_value());
    REQUIRE(sequence.mark_running().has_value());

    // Token 4 fills the existing second logical block.
    REQUIRE(sequence.append_token(4).has_value());
    REQUIRE(manager.ensure_capacity(sequence).has_value());
    CHECK(sequence.block_table().size() == 2);
    CHECK(manager.free_block_count() == 1);

    // Process token 4, then append token 5. Position 4 begins logical block 2.
    REQUIRE(sequence.schedule_tokens(1).has_value());
    REQUIRE(sequence.commit_scheduled_tokens().has_value());
    REQUIRE(sequence.append_token(5).has_value());
    REQUIRE(manager.ensure_capacity(sequence).has_value());
    REQUIRE(sequence.block_table().size() == 3);
    CHECK(sequence.block_table()[2] == 2);
    CHECK(manager.free_block_count() == 0);
}

TEST_CASE("release returns blocks and FIFO allocation reuses the oldest free ID")
{
    auto manager_result = block_manager::make(3, 2);
    auto first_result = seq::make(1, { 1, 2, 3 }, sampling_params {}, 2);
    REQUIRE(manager_result.has_value());
    REQUIRE(first_result.has_value());
    auto& manager = *manager_result;
    auto& first = *first_result;

    REQUIRE(manager.ensure_capacity(first).has_value());
    REQUIRE(manager.release(first).has_value());
    CHECK(first.block_table().empty());
    CHECK(first.cached_token_count() == 0);
    CHECK(manager.free_block_count() == 3);
    CHECK(manager.used_block_count() == 0);

    // Block 2 was never used and remained at the front while 0 and 1 were
    // released to the back.
    auto second_result = seq::make(2, { 9 }, sampling_params {}, 2);
    REQUIRE(second_result.has_value());
    auto& second = *second_result;
    REQUIRE(manager.ensure_capacity(second).has_value());
    REQUIRE(second.block_table().size() == 1);
    CHECK(second.block_table()[0] == 2);
}

TEST_CASE("release rejects running sequences without changing ownership")
{
    auto manager_result = block_manager::make(2, 2);
    auto sequence_result = seq::make(1, { 1, 2 }, sampling_params {}, 2);
    REQUIRE(manager_result.has_value());
    REQUIRE(sequence_result.has_value());
    auto& manager = *manager_result;
    auto& sequence = *sequence_result;

    REQUIRE(manager.ensure_capacity(sequence).has_value());
    REQUIRE(sequence.schedule_tokens(2).has_value());
    REQUIRE(sequence.commit_scheduled_tokens().has_value());
    REQUIRE(sequence.mark_running().has_value());

    auto released = manager.release(sequence);
    REQUIRE_FALSE(released.has_value());
    CHECK(released.error() == block_manager_errc::sequence_running);
    CHECK(sequence.status() == seq_status::running);
    CHECK(sequence.block_table().size() == 1);
    CHECK(manager.used_block_count() == 1);
}

TEST_CASE("release rejects scheduled work without changing ownership")
{
    auto manager_result = block_manager::make(2, 2);
    auto sequence_result = seq::make(1, { 1, 2 }, sampling_params {}, 2);
    REQUIRE(manager_result.has_value());
    REQUIRE(sequence_result.has_value());
    auto& manager = *manager_result;
    auto& sequence = *sequence_result;

    REQUIRE(manager.ensure_capacity(sequence).has_value());
    REQUIRE(sequence.schedule_tokens(1).has_value());

    auto released = manager.release(sequence);
    REQUIRE_FALSE(released.has_value());
    CHECK(released.error() == block_manager_errc::sequence_has_scheduled_work);
    CHECK(sequence.block_table().size() == 1);
    CHECK(sequence.scheduled_token_count() == 1);
    CHECK(manager.used_block_count() == 1);
}

TEST_CASE("manager rejects incompatible sequence geometry")
{
    auto manager_result = block_manager::make(4, 4);
    auto sequence_result = seq::make(1, { 1, 2 }, sampling_params {}, 2);
    REQUIRE(manager_result.has_value());
    REQUIRE(sequence_result.has_value());

    auto required = manager_result->additional_blocks_required(*sequence_result);
    REQUIRE_FALSE(required.has_value());
    CHECK(required.error() == block_manager_errc::incompatible_block_size);
}

TEST_CASE("capacity query reports a valid lack of free blocks")
{
    auto manager_result = block_manager::make(1, 2);
    auto first_result = seq::make(1, { 1 }, sampling_params {}, 2);
    auto second_result = seq::make(2, { 2 }, sampling_params {}, 2);
    REQUIRE(manager_result.has_value());
    REQUIRE(first_result.has_value());
    REQUIRE(second_result.has_value());
    auto& manager = *manager_result;

    REQUIRE(manager.ensure_capacity(*first_result).has_value());
    auto can_allocate = manager.can_ensure_capacity(*second_result);
    REQUIRE(can_allocate.has_value());
    CHECK_FALSE(*can_allocate);
}

TEST_CASE("existing block-table entries must reference live manager blocks")
{
    auto manager_result = block_manager::make(2, 2);
    REQUIRE(manager_result.has_value());
    auto& manager = *manager_result;

    auto free_reference_result = seq::make(1, { 1 }, sampling_params {}, 2);
    REQUIRE(free_reference_result.has_value());
    REQUIRE(free_reference_result->append_physical_block(0).has_value());
    auto free_reference = manager.additional_blocks_required(*free_reference_result);
    REQUIRE_FALSE(free_reference.has_value());
    CHECK(free_reference.error() == block_manager_errc::block_not_in_use);

    auto invalid_reference_result = seq::make(2, { 1 }, sampling_params {}, 2);
    REQUIRE(invalid_reference_result.has_value());
    REQUIRE(invalid_reference_result->append_physical_block(99).has_value());
    auto invalid_reference = manager.additional_blocks_required(*invalid_reference_result);
    REQUIRE_FALSE(invalid_reference.has_value());
    CHECK(invalid_reference.error() == block_manager_errc::invalid_block_id);
}
