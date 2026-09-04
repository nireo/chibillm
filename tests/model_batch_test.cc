#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <array>
#include <utility>

#include "model_batch.h"

using chibillm::batch_phase;
using chibillm::build_model_batch;
using chibillm::model_batch;
using chibillm::model_batch_errc;
using chibillm::model_batch_item;
using chibillm::position_id;
using chibillm::sampling_params;
using chibillm::scheduled_batch;
using chibillm::scheduler;
using chibillm::scheduler_config;
using chibillm::seq;
using chibillm::token_id;

namespace {

scheduler_config
test_config()
{
    return scheduler_config {
        .max_sequences = 4,
        .max_batch_tokens = 8,
        .kv_block_count = 8,
        .kv_block_size = 2,
        .eos_token = 99,
    };
}

} // namespace

TEST_CASE("prefill input contains the scheduled token slice and absolute positions")
{
    auto config = test_config();
    config.max_batch_tokens = 2;
    auto engine = scheduler::make(config);
    auto sequence = seq::make(1, { 10, 20, 30 }, sampling_params {}, 2);
    REQUIRE(engine.has_value());
    REQUIRE(sequence.has_value());
    REQUIRE(engine->add(std::move(*sequence)).has_value());

    auto scheduled = engine->schedule();
    REQUIRE(scheduled.has_value());
    auto built = build_model_batch(*scheduled, *engine);
    REQUIRE(built.has_value());

    CHECK(built->id == scheduled->id);
    CHECK(built->phase == batch_phase::prefill);
    CHECK(built->kv_block_size == 2);
    CHECK(built->tokens == std::vector<token_id> { 10, 20 });
    CHECK(built->positions == std::vector<position_id> { 0, 1 });
    REQUIRE(built->items.size() == 1);
    CHECK(built->items[0].id == 1);
    CHECK(built->items[0].token_offset == 0);
    CHECK(built->items[0].token_count == 2);
    CHECK(built->items[0].logits_index == 1);
    CHECK(built->items[0].block_table.size() == 2);

    const auto* unchanged = engine->find_sequence(1);
    REQUIRE(unchanged != nullptr);
    CHECK(unchanged->cached_token_count() == 0);
    CHECK(unchanged->scheduled_token_count() == 2);
}

TEST_CASE("later prefill chunks start at the cached prefix")
{
    auto config = test_config();
    config.max_batch_tokens = 2;
    auto engine = scheduler::make(config);
    auto sequence = seq::make(1, { 10, 20, 30 }, sampling_params {}, 2);
    REQUIRE(engine.has_value());
    REQUIRE(sequence.has_value());
    REQUIRE(engine->add(std::move(*sequence)).has_value());

    auto first = engine->schedule();
    REQUIRE(first.has_value());
    const std::array<token_id, 1> ignored_sample { 77 };
    REQUIRE(engine->complete(*first, ignored_sample).has_value());

    auto second = engine->schedule();
    REQUIRE(second.has_value());
    auto built = build_model_batch(*second, *engine);
    REQUIRE(built.has_value());
    CHECK(built->tokens == std::vector<token_id> { 30 });
    CHECK(built->positions == std::vector<position_id> { 2 });
    REQUIRE(built->items.size() == 1);
    CHECK(built->items[0].token_offset == 0);
    CHECK(built->items[0].token_count == 1);
    CHECK(built->items[0].logits_index == 0);
}

TEST_CASE("decode input contains the previous model sample")
{
    auto engine = scheduler::make(test_config());
    auto sequence = seq::make(1, { 10, 20 }, sampling_params {}, 2);
    REQUIRE(engine.has_value());
    REQUIRE(sequence.has_value());
    REQUIRE(engine->add(std::move(*sequence)).has_value());

    auto prefill = engine->schedule();
    REQUIRE(prefill.has_value());
    const std::array<token_id, 1> first_sample { 30 };
    REQUIRE(engine->complete(*prefill, first_sample).has_value());

    auto decode = engine->schedule();
    REQUIRE(decode.has_value());
    auto built = build_model_batch(*decode, *engine);
    REQUIRE(built.has_value());
    CHECK(built->phase == batch_phase::decode);
    CHECK(built->tokens == std::vector<token_id> { 30 });
    CHECK(built->positions == std::vector<position_id> { 2 });
    REQUIRE(built->items.size() == 1);
    CHECK(built->items[0].logits_index == 0);
    CHECK(built->items[0].block_table.size() == 2);
}

TEST_CASE("ragged prefill is flattened in scheduled order")
{
    auto config = test_config();
    config.max_sequences = 2;
    config.max_batch_tokens = 3;
    auto engine = scheduler::make(config);
    auto first = seq::make(1, { 10, 11 }, sampling_params {}, 2);
    auto second = seq::make(2, { 20, 21 }, sampling_params {}, 2);
    REQUIRE(engine.has_value());
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(engine->add(std::move(*first)).has_value());
    REQUIRE(engine->add(std::move(*second)).has_value());

    auto scheduled = engine->schedule();
    REQUIRE(scheduled.has_value());
    auto built = build_model_batch(*scheduled, *engine);
    REQUIRE(built.has_value());

    CHECK(built->tokens == std::vector<token_id> { 10, 11, 20 });
    CHECK(built->positions == std::vector<position_id> { 0, 1, 0 });
    REQUIRE(built->items.size() == 2);
    CHECK(built->items[0].id == 1);
    CHECK(built->items[0].token_offset == 0);
    CHECK(built->items[0].token_count == 2);
    CHECK(built->items[0].logits_index == 1);
    CHECK(built->items[1].id == 2);
    CHECK(built->items[1].token_offset == 2);
    CHECK(built->items[1].token_count == 1);
    CHECK(built->items[1].logits_index == 2);
}

TEST_CASE("model batch rejects invalid scheduled batches")
{
    auto engine = scheduler::make(test_config());
    REQUIRE(engine.has_value());

    SUBCASE("empty batch")
    {
        const scheduled_batch scheduled { .id = 1, .phase = batch_phase::prefill, .items = {} };
        CHECK(build_model_batch(scheduled, *engine).error() == model_batch_errc::empty_batch);
    }

    SUBCASE("reservation mismatch and duplicate ids")
    {
        auto sequence = seq::make(1, { 10 }, sampling_params {}, 2);
        REQUIRE(sequence.has_value());
        REQUIRE(engine->add(std::move(*sequence)).has_value());

        auto scheduled = engine->schedule();
        REQUIRE(scheduled.has_value());

        auto mismatched = *scheduled;
        ++mismatched.items[0].token_count;
        CHECK(build_model_batch(mismatched, *engine).error()
              == model_batch_errc::scheduled_token_count_mismatch);

        auto duplicated = *scheduled;
        duplicated.items.push_back(duplicated.items.front());
        CHECK(build_model_batch(duplicated, *engine).error()
              == model_batch_errc::duplicate_sequence_id);

        const auto* unchanged = engine->find_sequence(1);
        REQUIRE(unchanged != nullptr);
        CHECK(unchanged->cached_token_count() == 0);
        CHECK(unchanged->scheduled_token_count() == 1);
    }
}
