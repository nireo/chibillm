#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <array>
#include <utility>
#include <vector>

#include "scheduler.h"

using chibillm::batch_phase;
using chibillm::finish_reason;
using chibillm::sampling_params;
using chibillm::scheduled_batch;
using chibillm::scheduled_item;
using chibillm::scheduler;
using chibillm::scheduler_config;
using chibillm::scheduler_errc;
using chibillm::seq;
using chibillm::seq_status;
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

TEST_CASE("scheduler construction validates limits and exposes cache geometry")
{
    auto config = test_config();

    config.max_sequences = 0;
    auto zero_sequences = scheduler::make(config);
    REQUIRE_FALSE(zero_sequences.has_value());
    CHECK(zero_sequences.error() == scheduler_errc::invalid_max_sequences);

    config = test_config();
    config.max_batch_tokens = 0;
    auto zero_tokens = scheduler::make(config);
    REQUIRE_FALSE(zero_tokens.has_value());
    CHECK(zero_tokens.error() == scheduler_errc::invalid_max_batch_tokens);

    config = test_config();
    config.kv_block_count = 0;
    auto zero_blocks = scheduler::make(config);
    REQUIRE_FALSE(zero_blocks.has_value());
    CHECK(zero_blocks.error() == scheduler_errc::invalid_kv_block_count);

    config = test_config();
    config.kv_block_size = 0;
    auto zero_block_size = scheduler::make(config);
    REQUIRE_FALSE(zero_block_size.has_value());
    CHECK(zero_block_size.error() == scheduler_errc::invalid_kv_block_size);

    auto result = scheduler::make(test_config());
    REQUIRE(result.has_value());
    CHECK(result->sequence_count() == 0);
    CHECK(result->waiting_count() == 0);
    CHECK(result->running_count() == 0);
    CHECK(result->is_finished());
    CHECK(result->cache().block_count() == 8);
    CHECK(result->cache().block_size() == 2);
}

TEST_CASE("scheduled batch accessors summarize items")
{
    const scheduled_batch batch {
        .id = 7,
        .phase = batch_phase::prefill,
        .items = {
            scheduled_item { .id = 1, .token_count = 3 },
            scheduled_item { .id = 2, .token_count = 2 },
        },
    };

    CHECK_FALSE(batch.empty());
    CHECK(batch.token_count() == 5);
}

TEST_CASE("completion without an active batch is rejected")
{
    auto scheduler_result = scheduler::make(test_config());
    REQUIRE(scheduler_result.has_value());

    const scheduled_batch phantom {
        .id = 1,
        .phase = batch_phase::prefill,
        .items = { scheduled_item { .id = 1, .token_count = 1 } },
    };
    const std::array<token_id, 1> sample { 42 };

    auto completed = scheduler_result->complete(phantom, sample);
    REQUIRE_FALSE(completed.has_value());
    CHECK(completed.error() == scheduler_errc::no_batch_in_flight);
    CHECK_FALSE(scheduler_result->has_in_flight_batch());
}

TEST_CASE("add admits one pristine waiting sequence and rejects a duplicate ID")
{
    auto scheduler_result = scheduler::make(test_config());
    auto first = seq::make(10, { 1, 2 }, sampling_params {}, 2);
    REQUIRE(scheduler_result.has_value());
    REQUIRE(first.has_value());
    auto& engine = *scheduler_result;

    REQUIRE(engine.add(std::move(*first)).has_value());
    CHECK(engine.sequence_count() == 1);
    CHECK(engine.waiting_count() == 1);
    CHECK(engine.running_count() == 0);
    REQUIRE(engine.find_sequence(10) != nullptr);

    auto duplicate = seq::make(10, { 8 }, sampling_params {}, 2);
    REQUIRE(duplicate.has_value());
    auto added_twice = engine.add(std::move(*duplicate));
    REQUIRE_FALSE(added_twice.has_value());
    CHECK(added_twice.error() == scheduler_errc::duplicate_sequence_id);
    CHECK(engine.sequence_count() == 1);
    CHECK(engine.waiting_count() == 1);
}

TEST_CASE("prefill can be chunked and only the final chunk appends a sample")
{
    auto config = test_config();
    config.max_batch_tokens = 2;
    auto scheduler_result = scheduler::make(config);
    auto sequence = seq::make(1, { 10, 20, 30 }, sampling_params {}, 2);
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());

    auto first_batch = engine.schedule();
    REQUIRE(first_batch.has_value());
    CHECK(first_batch->phase == batch_phase::prefill);
    REQUIRE(first_batch->items.size() == 1);
    CHECK(first_batch->items[0].token_count == 2);

    auto overlapping = engine.schedule();
    REQUIRE_FALSE(overlapping.has_value());
    CHECK(overlapping.error() == scheduler_errc::batch_in_flight);

    const std::array<token_id, 1> intermediate_sample { 777 };
    REQUIRE(engine.complete(*first_batch, intermediate_sample).has_value());
    const auto* after_first = engine.find_sequence(1);
    REQUIRE(after_first != nullptr);
    CHECK(after_first->cached_token_count() == 2);
    CHECK(after_first->token_count() == 3);
    CHECK(after_first->status() == seq_status::waiting);

    auto final_batch = engine.schedule();
    REQUIRE(final_batch.has_value());
    CHECK(final_batch->phase == batch_phase::prefill);
    REQUIRE(final_batch->items.size() == 1);
    CHECK(final_batch->items[0].token_count == 1);

    const std::array<token_id, 1> final_sample { 40 };
    REQUIRE(engine.complete(*final_batch, final_sample).has_value());
    const auto* completed_prompt = engine.find_sequence(1);
    REQUIRE(completed_prompt != nullptr);
    CHECK(completed_prompt->status() == seq_status::running);
    CHECK(completed_prompt->cached_token_count() == 3);
    CHECK(completed_prompt->completion_token_count() == 1);
    CHECK(completed_prompt->last_token() == 40);
    CHECK(engine.waiting_count() == 0);
    CHECK(engine.running_count() == 1);
}

TEST_CASE("decode commits the old sample and creates the next one-token cache gap")
{
    auto scheduler_result = scheduler::make(test_config());
    auto sequence = seq::make(1, { 10, 20 }, sampling_params {}, 2);
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());

    auto prefill = engine.schedule();
    REQUIRE(prefill.has_value());
    const std::array<token_id, 1> first_sample { 30 };
    REQUIRE(engine.complete(*prefill, first_sample).has_value());

    const auto* before_decode = engine.find_sequence(1);
    REQUIRE(before_decode != nullptr);
    CHECK(before_decode->token_count() == 3);
    CHECK(before_decode->cached_token_count() == 2);

    auto decode = engine.schedule();
    REQUIRE(decode.has_value());
    CHECK(decode->phase == batch_phase::decode);
    REQUIRE(decode->items.size() == 1);
    CHECK(decode->items[0].token_count == 1);

    const std::array<token_id, 1> next_sample { 40 };
    REQUIRE(engine.complete(*decode, next_sample).has_value());
    const auto* after_decode = engine.find_sequence(1);
    REQUIRE(after_decode != nullptr);
    CHECK(after_decode->token_count() == 4);
    CHECK(after_decode->cached_token_count() == 3);
    CHECK(after_decode->uncached_token_count() == 1);
    CHECK(after_decode->last_token() == 40);
}

TEST_CASE("EOS on the first sample finishes the sequence and releases cache blocks")
{
    auto scheduler_result = scheduler::make(test_config());
    auto sequence = seq::make(1, { 10, 20 }, sampling_params {}, 2);
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());

    auto prefill = engine.schedule();
    REQUIRE(prefill.has_value());
    const std::array<token_id, 1> eos { 99 };
    REQUIRE(engine.complete(*prefill, eos).has_value());

    const auto* finished = engine.find_sequence(1);
    REQUIRE(finished != nullptr);
    CHECK(finished->status() == seq_status::finished);
    CHECK(finished->reason() == finish_reason::eos);
    CHECK(finished->block_table().empty());
    CHECK(engine.cache().used_block_count() == 0);
    CHECK(engine.is_finished());
}

TEST_CASE("maximum new-token limit is evaluated after appending the sample")
{
    auto scheduler_result = scheduler::make(test_config());
    const sampling_params params { .max_new_tokens = 1 };
    auto sequence = seq::make(1, { 10 }, params, 2);
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());

    auto prefill = engine.schedule();
    REQUIRE(prefill.has_value());
    const std::array<token_id, 1> sample { 42 };
    REQUIRE(engine.complete(*prefill, sample).has_value());

    const auto* finished = engine.find_sequence(1);
    REQUIRE(finished != nullptr);
    CHECK(finished->reason() == finish_reason::len_limit);
    CHECK(finished->completion_token_count() == 1);
    CHECK(engine.is_finished());
}

TEST_CASE("cancel releases a running sequence and allows it to be retired")
{
    auto scheduler_result = scheduler::make(test_config());
    auto sequence = seq::make(1, { 10, 20 }, sampling_params {}, 2);
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());

    auto prefill = engine.schedule();
    REQUIRE(prefill.has_value());
    const std::array<token_id, 1> sample { 30 };
    REQUIRE(engine.complete(*prefill, sample).has_value());
    CHECK(engine.cache().used_block_count() == 1);

    REQUIRE(engine.cancel(1).has_value());
    const auto* cancelled = engine.find_sequence(1);
    REQUIRE(cancelled != nullptr);
    CHECK(cancelled->reason() == finish_reason::cancelled);
    CHECK(engine.cache().used_block_count() == 0);
    CHECK(engine.is_finished());

    REQUIRE(engine.remove(1).has_value());
    CHECK(engine.find_sequence(1) == nullptr);
    CHECK(engine.sequence_count() == 0);
}

TEST_CASE("completion validation leaves an in-flight reservation untouched")
{
    auto scheduler_result = scheduler::make(test_config());
    auto sequence = seq::make(1, { 10 }, sampling_params {}, 2);
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());
    auto batch = engine.schedule();
    REQUIRE(batch.has_value());

    const std::array<token_id, 0> no_samples {};
    auto completed = engine.complete(*batch, no_samples);
    REQUIRE_FALSE(completed.has_value());
    CHECK(completed.error() == scheduler_errc::result_count_mismatch);
    CHECK(engine.has_in_flight_batch());
    const auto* unchanged = engine.find_sequence(1);
    REQUIRE(unchanged != nullptr);
    CHECK(unchanged->scheduled_token_count() == 1);
    CHECK(unchanged->cached_token_count() == 0);
}

TEST_CASE("abort cancels reservations and retains allocated cache blocks")
{
    auto scheduler_result = scheduler::make(test_config());
    auto sequence = seq::make(1, { 10, 20, 30 }, sampling_params {}, 2);
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());

    auto batch = engine.schedule();
    REQUIRE(batch.has_value());
    CHECK(engine.cache().used_block_count() == 2);

    REQUIRE(engine.abort(*batch).has_value());
    CHECK_FALSE(engine.has_in_flight_batch());
    CHECK(engine.waiting_count() == 1);
    CHECK(engine.running_count() == 0);
    CHECK(engine.cache().used_block_count() == 2);

    const auto* unchanged = engine.find_sequence(1);
    REQUIRE(unchanged != nullptr);
    CHECK(unchanged->cached_token_count() == 0);
    CHECK(unchanged->scheduled_token_count() == 0);
    CHECK(unchanged->block_table().size() == 2);

    auto retried = engine.schedule();
    REQUIRE(retried.has_value());
    REQUIRE(retried->items.size() == 1);
    CHECK(retried->items[0].id == batch->items[0].id);
    CHECK(retried->items[0].token_count == batch->items[0].token_count);
}

TEST_CASE("abort validates the active batch before changing reservations")
{
    auto scheduler_result = scheduler::make(test_config());
    auto sequence = seq::make(1, { 10 }, sampling_params {}, 2);
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());

    auto batch = engine.schedule();
    REQUIRE(batch.has_value());

    auto wrong_batch = *batch;
    ++wrong_batch.id;
    auto rejected = engine.abort(wrong_batch);
    REQUIRE_FALSE(rejected.has_value());
    CHECK(rejected.error() == scheduler_errc::batch_id_mismatch);
    CHECK(engine.has_in_flight_batch());

    const auto* unchanged = engine.find_sequence(1);
    REQUIRE(unchanged != nullptr);
    CHECK(unchanged->scheduled_token_count() == 1);

    REQUIRE(engine.abort(*batch).has_value());
}

TEST_CASE("waiting prefill work is chosen before existing decode work")
{
    auto scheduler_result = scheduler::make(test_config());
    auto first = seq::make(1, { 10 }, sampling_params {}, 2);
    REQUIRE(scheduler_result.has_value());
    REQUIRE(first.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*first)).has_value());

    auto first_prefill = engine.schedule();
    REQUIRE(first_prefill.has_value());
    const std::array<token_id, 1> first_sample { 11 };
    REQUIRE(engine.complete(*first_prefill, first_sample).has_value());

    auto newcomer = seq::make(2, { 20 }, sampling_params {}, 2);
    REQUIRE(newcomer.has_value());
    REQUIRE(engine.add(std::move(*newcomer)).has_value());

    auto next = engine.schedule();
    REQUIRE(next.has_value());
    CHECK(next->phase == batch_phase::prefill);
    REQUIRE(next->items.size() == 1);
    CHECK(next->items[0].id == 2);
}

TEST_CASE("cache exhaustion reports an error without partially scheduling")
{
    auto config = test_config();
    config.kv_block_count = 1;
    auto scheduler_result = scheduler::make(config);
    auto sequence = seq::make(1, { 10, 20, 30 }, sampling_params {}, 2);
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());

    auto batch = engine.schedule();
    REQUIRE_FALSE(batch.has_value());
    CHECK(batch.error() == scheduler_errc::cache_capacity_exhausted);
    CHECK_FALSE(engine.has_in_flight_batch());
    CHECK(engine.cache().used_block_count() == 0);
    const auto* unchanged = engine.find_sequence(1);
    REQUIRE(unchanged != nullptr);
    CHECK(unchanged->scheduled_token_count() == 0);
    CHECK(unchanged->block_table().empty());
    CHECK(unchanged->status() == seq_status::waiting);
}
