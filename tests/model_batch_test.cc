#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <array>
#include <utility>

#include "model_batch.h"

using chibillm::batch_phase;
using chibillm::build_model_batch;
using chibillm::generation_params;
using chibillm::model_batch;
using chibillm::model_batch_errc;
using chibillm::model_batch_item;
using chibillm::position_id;
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

TEST_CASE("model batch builds chunked prefill and subsequent decode batches")
{
    auto config = test_config();
    config.max_batch_tokens = 2;
    auto engine = scheduler::make(config);
    auto sequence = seq::make(1, { 10, 20, 30 }, generation_params {});
    REQUIRE(engine.has_value());
    REQUIRE(sequence.has_value());
    REQUIRE(engine->add(std::move(*sequence)).has_value());

    // Chunk 1: first prefill slice [10, 20]
    auto first = engine->schedule();
    REQUIRE(first.has_value());
    auto first_batch = build_model_batch(*first, *engine);
    REQUIRE(first_batch.has_value());

    CHECK(first_batch->id == first->id);
    CHECK(first_batch->phase == batch_phase::prefill);
    CHECK(first_batch->kv_block_size == 2);
    CHECK(first_batch->tokens == std::vector<token_id> { 10, 20 });
    CHECK(first_batch->positions == std::vector<position_id> { 0, 1 });
    REQUIRE(first_batch->items.size() == 1);
    CHECK(first_batch->items[0].id == 1);
    CHECK(first_batch->items[0].token_offset == 0);
    CHECK(first_batch->items[0].token_count == 2);
    CHECK_FALSE(first_batch->items[0].logits_index.has_value());
    CHECK(first_batch->items[0].block_table.size() == 2);

    const auto* unchanged = engine->find_sequence(1);
    REQUIRE(unchanged != nullptr);
    CHECK(unchanged->processed_token_count() == 0);
    CHECK(unchanged->scheduled_token_count() == 2);

    // Chunk 2: remaining prefill slice [30] starting at cached prefix position 2
    const std::array<token_id, 0> ignored_sample {};
    REQUIRE(engine->complete(first->id, ignored_sample).has_value());

    auto second = engine->schedule();
    REQUIRE(second.has_value());
    auto second_batch = build_model_batch(*second, *engine);
    REQUIRE(second_batch.has_value());
    CHECK(second_batch->tokens == std::vector<token_id> { 30 });
    CHECK(second_batch->positions == std::vector<position_id> { 2 });
    REQUIRE(second_batch->items.size() == 1);
    CHECK(second_batch->items[0].token_offset == 0);
    CHECK(second_batch->items[0].token_count == 1);
    CHECK(second_batch->items[0].logits_index == 0);

    // Decode: subsequent step feeds the previous sample (30) at position 3
    const std::array<token_id, 1> decode_sample { 40 };
    REQUIRE(engine->complete(second->id, decode_sample).has_value());

    auto decode = engine->schedule();
    REQUIRE(decode.has_value());
    auto decode_batch = build_model_batch(*decode, *engine);
    REQUIRE(decode_batch.has_value());
    CHECK(decode_batch->phase == batch_phase::decode);
    CHECK(decode_batch->tokens == std::vector<token_id> { 40 });
    CHECK(decode_batch->positions == std::vector<position_id> { 3 });
    REQUIRE(decode_batch->items.size() == 1);
    CHECK(decode_batch->items[0].logits_index == 0);
    CHECK(decode_batch->items[0].block_table.size() == 2);
}

TEST_CASE("ragged prefill is flattened in scheduled order")
{
    auto config = test_config();
    config.max_sequences = 2;
    config.max_batch_tokens = 3;
    auto engine = scheduler::make(config);
    auto first = seq::make(1, { 10, 11 }, generation_params {});
    auto second = seq::make(2, { 20, 21 }, generation_params {});
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
    CHECK_FALSE(built->items[1].logits_index.has_value());
    auto updates = engine->complete(scheduled->id, std::array<token_id, 1> { 42 });
    REQUIRE(updates.has_value());
    REQUIRE(updates->size() == 1);
    CHECK(updates->front().id == 1);
    CHECK(engine->find_sequence(2)->completion_token_count() == 0);
    CHECK(engine->find_sequence(2)->processed_token_count() == 1);
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
        auto sequence = seq::make(1, { 10 }, generation_params {});
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
        CHECK(unchanged->processed_token_count() == 0);
        CHECK(unchanged->scheduled_token_count() == 1);
    }
}
