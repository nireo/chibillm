#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <utility>
#include <vector>

#include "scheduler.h"

using chibillm::batch_phase;
using chibillm::finish_reason;
using chibillm::generation_params;
using chibillm::scheduled_batch;
using chibillm::scheduled_item;
using chibillm::scheduler;
using chibillm::scheduler_config;
using chibillm::scheduler_errc;
using chibillm::seq;
using chibillm::seq_status;
using chibillm::token_id;

namespace {
const chibillm::block_manager&
paged_state(const scheduler& scheduler)
{
    return dynamic_cast<const chibillm::block_manager&>(scheduler.state());
}

scheduler_config
test_config()
{
    return scheduler_config {
        .max_sequences = 4,
        .max_batch_tokens = 8,
        .kv_block_count = 8,
        .kv_block_size = 2,
    };
}

void
commit(scheduler& engine, const scheduled_batch& batch)
{
    std::vector<token_id> samples;
    for (const auto& item : batch.items)
        if (item.sample)
            samples.push_back(42);
    REQUIRE(engine.complete(batch.id, samples));
}

} // namespace

TEST_CASE("scheduler construction validates limits and exposes cache geometry")
{
    auto config = test_config();

    config.max_sequences = 0;
    auto zero_sequences = scheduler::make(config, 99);
    REQUIRE_FALSE(zero_sequences.has_value());
    CHECK(zero_sequences.error() == scheduler_errc::invalid_max_sequences);

    config = test_config();
    config.max_batch_tokens = 0;
    auto zero_tokens = scheduler::make(config, 99);
    REQUIRE_FALSE(zero_tokens.has_value());
    CHECK(zero_tokens.error() == scheduler_errc::invalid_max_batch_tokens);

    config = test_config();
    config.prefill_chunk_tokens = 0;
    CHECK(scheduler::make(config, 99).error() == scheduler_errc::invalid_prefill_chunk_tokens);

    config = test_config();
    config.kv_block_count = 0;
    auto zero_blocks = scheduler::make(config, 99);
    REQUIRE_FALSE(zero_blocks.has_value());
    CHECK(zero_blocks.error() == scheduler_errc::invalid_kv_block_count);

    config = test_config();
    config.kv_block_size = 0;
    auto zero_block_size = scheduler::make(config, 99);
    REQUIRE_FALSE(zero_block_size.has_value());
    CHECK(zero_block_size.error() == scheduler_errc::invalid_kv_block_size);

    auto result = scheduler::make(test_config(), 99);
    REQUIRE(result.has_value());
    CHECK(result->sequence_count() == 0);
    CHECK(result->waiting_count() == 0);
    CHECK(result->running_count() == 0);
    CHECK(result->is_finished());
    CHECK(paged_state(*result).block_count() == 8);
    CHECK(paged_state(*result).block_size() == 2);
}

TEST_CASE("completion without an active batch is rejected")
{
    auto scheduler_result = scheduler::make(test_config(), 99);
    REQUIRE(scheduler_result.has_value());

    const scheduled_batch phantom {
        .id = 1,
        .phase = batch_phase::prefill,
        .items = { scheduled_item { .id = 1, .token_count = 1 } },
    };
    const std::array<token_id, 1> sample { 42 };

    auto completed = scheduler_result->complete(phantom.id, sample);
    REQUIRE_FALSE(completed.has_value());
    CHECK(completed.error() == scheduler_errc::no_batch_in_flight);
    CHECK_FALSE(scheduler_result->has_in_flight_batch());
}

TEST_CASE("add admits one pristine waiting sequence and rejects a duplicate ID")
{
    auto scheduler_result = scheduler::make(test_config(), 99);
    auto first = seq::make(10, { 1, 2 }, generation_params { .max_new_tokens = 4 });
    REQUIRE(scheduler_result.has_value());
    REQUIRE(first.has_value());
    auto& engine = *scheduler_result;

    REQUIRE(engine.add(std::move(*first)).has_value());
    CHECK(engine.sequence_count() == 1);
    CHECK(engine.waiting_count() == 1);
    CHECK(engine.running_count() == 0);
    REQUIRE(engine.find_sequence(10) != nullptr);

    auto duplicate = seq::make(10, { 8 }, generation_params { .max_new_tokens = 4 });
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
    auto scheduler_result = scheduler::make(config, 99);
    auto sequence = seq::make(1, { 10, 20, 30 }, generation_params { .max_new_tokens = 4 });
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

    const std::array<token_id, 0> intermediate_sample {};
    REQUIRE(engine.complete(first_batch->id, intermediate_sample).has_value());
    const auto* after_first = engine.find_sequence(1);
    REQUIRE(after_first != nullptr);
    CHECK(after_first->processed_token_count() == 2);
    CHECK(after_first->token_count() == 3);
    CHECK(after_first->status() == seq_status::waiting);

    auto final_batch = engine.schedule();
    REQUIRE(final_batch.has_value());
    CHECK(final_batch->phase == batch_phase::prefill);
    REQUIRE(final_batch->items.size() == 1);
    CHECK(final_batch->items[0].token_count == 1);

    const std::array<token_id, 1> final_sample { 40 };
    REQUIRE(engine.complete(final_batch->id, final_sample).has_value());
    const auto* completed_prompt = engine.find_sequence(1);
    REQUIRE(completed_prompt != nullptr);
    CHECK(completed_prompt->status() == seq_status::running);
    CHECK(completed_prompt->processed_token_count() == 3);
    CHECK(completed_prompt->completion_token_count() == 1);
    CHECK(completed_prompt->last_token() == 40);
    CHECK(engine.waiting_count() == 0);
    CHECK(engine.running_count() == 1);
}

TEST_CASE("decode commits the old sample and creates the next one-token cache gap")
{
    auto scheduler_result = scheduler::make(test_config(), 99);
    auto sequence = seq::make(1, { 10, 20 }, generation_params { .max_new_tokens = 4 });
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());

    auto prefill = engine.schedule();
    REQUIRE(prefill.has_value());
    const std::array<token_id, 1> first_sample { 30 };
    REQUIRE(engine.complete(prefill->id, first_sample).has_value());

    const auto* before_decode = engine.find_sequence(1);
    REQUIRE(before_decode != nullptr);
    CHECK(before_decode->token_count() == 3);
    CHECK(before_decode->processed_token_count() == 2);

    auto decode = engine.schedule();
    REQUIRE(decode.has_value());
    CHECK(decode->phase == batch_phase::decode);
    REQUIRE(decode->items.size() == 1);
    CHECK(decode->items[0].token_count == 1);

    const std::array<token_id, 1> next_sample { 40 };
    REQUIRE(engine.complete(decode->id, next_sample).has_value());
    const auto* after_decode = engine.find_sequence(1);
    REQUIRE(after_decode != nullptr);
    CHECK(after_decode->token_count() == 4);
    CHECK(after_decode->processed_token_count() == 3);
    CHECK(after_decode->unprocessed_token_count() == 1);
    CHECK(after_decode->last_token() == 40);
}

TEST_CASE("sequences finish on EOS or length limit and release cache blocks")
{
    auto check_finish = [](generation_params params, token_id sample,
                           finish_reason expected_reason) {
        auto scheduler_result = scheduler::make(test_config(), 99);
        auto sequence = seq::make(1, { 10, 20 }, params);
        REQUIRE(scheduler_result.has_value());
        REQUIRE(sequence.has_value());
        auto& engine = *scheduler_result;
        REQUIRE(engine.add(std::move(*sequence)).has_value());

        auto prefill = engine.schedule();
        REQUIRE(prefill.has_value());
        REQUIRE(engine.complete(prefill->id, std::array<token_id, 1> { sample }).has_value());

        const auto* finished = engine.find_sequence(1);
        REQUIRE(finished != nullptr);
        CHECK(finished->status() == seq_status::finished);
        CHECK(finished->reason() == expected_reason);
        CHECK(engine.state().resources(1).blocks.empty());
        CHECK(paged_state(engine).used_block_count() == 0);
        CHECK(engine.is_finished());
    };

    check_finish(generation_params { .max_new_tokens = 4 }, 99, finish_reason::eos);
    check_finish(generation_params { .max_new_tokens = 1 }, 42, finish_reason::len_limit);
}

TEST_CASE("cancel releases a running sequence and allows it to be retired")
{
    auto scheduler_result = scheduler::make(test_config(), 99);
    auto sequence = seq::make(1, { 10, 20 }, generation_params { .max_new_tokens = 4 });
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());

    auto prefill = engine.schedule();
    REQUIRE(prefill.has_value());
    const std::array<token_id, 1> sample { 30 };
    REQUIRE(engine.complete(prefill->id, sample).has_value());
    CHECK(paged_state(engine).used_block_count() == 1);

    REQUIRE(engine.cancel(1).has_value());
    const auto* cancelled = engine.find_sequence(1);
    REQUIRE(cancelled != nullptr);
    CHECK(cancelled->reason() == finish_reason::cancelled);
    CHECK(paged_state(engine).used_block_count() == 0);
    CHECK(engine.is_finished());

    REQUIRE(engine.remove(1).has_value());
    CHECK(engine.find_sequence(1) == nullptr);
    CHECK(engine.sequence_count() == 0);
}

TEST_CASE("completion validation leaves an in-flight reservation untouched")
{
    auto scheduler_result = scheduler::make(test_config(), 99);
    auto sequence = seq::make(1, { 10 }, generation_params { .max_new_tokens = 4 });
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());
    auto batch = engine.schedule();
    REQUIRE(batch.has_value());

    const std::array<token_id, 0> no_samples {};
    auto completed = engine.complete(batch->id, no_samples);
    REQUIRE_FALSE(completed.has_value());
    CHECK(completed.error() == scheduler_errc::result_count_mismatch);
    CHECK(engine.has_in_flight_batch());
    const auto* unchanged = engine.find_sequence(1);
    REQUIRE(unchanged != nullptr);
    CHECK(unchanged->scheduled_token_count() == 1);
    CHECK(unchanged->processed_token_count() == 0);
}

TEST_CASE("abort validates active batch and restores reservations")
{
    auto scheduler_result = scheduler::make(test_config(), 99);
    auto sequence = seq::make(1, { 10, 20, 30 }, generation_params { .max_new_tokens = 4 });
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*sequence)).has_value());

    auto batch = engine.schedule();
    REQUIRE(batch.has_value());
    CHECK(paged_state(engine).used_block_count() == 2);

    auto wrong_batch = *batch;
    ++wrong_batch.id;
    CHECK(engine.abort(wrong_batch.id).error() == scheduler_errc::batch_id_mismatch);
    CHECK(engine.has_in_flight_batch());

    REQUIRE(engine.abort(batch->id).has_value());
    CHECK_FALSE(engine.has_in_flight_batch());
    CHECK(engine.waiting_count() == 1);
    CHECK(engine.running_count() == 0);
    CHECK(paged_state(engine).used_block_count() == 2);

    const auto* unchanged = engine.find_sequence(1);
    REQUIRE(unchanged != nullptr);
    CHECK(unchanged->processed_token_count() == 0);
    CHECK(unchanged->scheduled_token_count() == 0);
    CHECK(engine.state().resources(1).blocks.size() == 2);

    auto retried = engine.schedule();
    REQUIRE(retried.has_value());
    REQUIRE(retried->items.size() == 1);
    CHECK(retried->items[0].id == batch->items[0].id);
    CHECK(retried->items[0].token_count == batch->items[0].token_count);
}

TEST_CASE("new prefill waits for an existing decode round")
{
    auto scheduler_result = scheduler::make(test_config(), 99);
    auto first = seq::make(1, { 10 }, generation_params { .max_new_tokens = 4 });
    REQUIRE(scheduler_result.has_value());
    REQUIRE(first.has_value());
    auto& engine = *scheduler_result;
    REQUIRE(engine.add(std::move(*first)).has_value());

    auto first_prefill = engine.schedule();
    REQUIRE(first_prefill.has_value());
    const std::array<token_id, 1> first_sample { 11 };
    REQUIRE(engine.complete(first_prefill->id, first_sample).has_value());

    auto newcomer = seq::make(2, { 20 }, generation_params { .max_new_tokens = 4 });
    REQUIRE(newcomer.has_value());
    REQUIRE(engine.add(std::move(*newcomer)).has_value());

    auto next = engine.schedule();
    REQUIRE(next.has_value());
    CHECK(next->phase == batch_phase::decode);
    REQUIRE(next->items.size() == 1);
    CHECK(next->items[0].id == 1);
    REQUIRE(engine.complete(next->id, std::array<token_id, 1> { 12 }));
    next = engine.schedule();
    REQUIRE(next);
    CHECK(next->phase == batch_phase::prefill);
    CHECK(next->items[0].id == 2);
}

TEST_CASE("an oversized request is rejected without allocating or consuming it")
{
    auto config = test_config();
    config.kv_block_count = 1;
    auto scheduler_result = scheduler::make(config, 99);
    auto sequence = seq::make(1, { 10, 20, 30 }, generation_params { .max_new_tokens = 4 });
    REQUIRE(scheduler_result.has_value());
    REQUIRE(sequence.has_value());
    auto& engine = *scheduler_result;
    auto admitted = engine.try_add(*sequence);
    REQUIRE(admitted);
    CHECK(*admitted == chibillm::admission_result::too_large);
    CHECK(sequence->prompt_token_count() == 3);
    CHECK_FALSE(engine.has_in_flight_batch());
    CHECK(paged_state(engine).used_block_count() == 0);
    CHECK(engine.find_sequence(1) == nullptr);
    CHECK(engine.state().resources(1).blocks.empty());
    CHECK(sequence->status() == seq_status::waiting);
}

TEST_CASE("contended prompt chunks rotate only after successful completion")
{
    auto config = test_config();
    config.kv_block_count = 32;
    config.max_batch_tokens = 2;
    config.prefill_chunk_tokens = 1;
    auto engine = scheduler::make(config, 99);
    REQUIRE(engine);
    for (const auto id : { 1, 2, 3, 4 }) {
        auto sequence = seq::make(id, std::vector<token_id>(5, 10), { .max_new_tokens = 2 });
        REQUIRE(sequence);
        REQUIRE(engine->add(std::move(*sequence)));
    }
    auto batch = engine->schedule();
    REQUIRE(batch);
    REQUIRE(batch->items.size() == 2);
    CHECK(batch->items[0].id == 1);
    CHECK(batch->items[1].id == 2);
    CHECK(batch->items[0].token_count == 1);
    REQUIRE(engine->abort(batch->id));
    auto retry = engine->schedule();
    REQUIRE(retry);
    CHECK(retry->items[0].id == 1);
    CHECK(retry->items[1].id == 2);
    commit(*engine, *retry);
    batch = engine->schedule();
    REQUIRE(batch);
    CHECK(batch->items[0].id == 3);
    CHECK(batch->items[1].id == 4);
    commit(*engine, *batch);
    for (const auto id : { 1, 2, 3, 4 })
        CHECK(engine->find_sequence(id)->processed_token_count() == 1);
}

TEST_CASE("a new arrival can shorten an aborted prefill retry without shrinking its pages")
{
    auto config = test_config();
    config.max_batch_tokens = 4;
    config.prefill_chunk_tokens = 1;
    auto engine = scheduler::make(config, 99);
    REQUIRE(engine);
    auto first = seq::make(1, std::vector<token_id>(5, 10), { .max_new_tokens = 2 });
    REQUIRE(first);
    REQUIRE(engine->add(std::move(*first)));
    auto batch = engine->schedule();
    REQUIRE(batch);
    CHECK(batch->items[0].token_count == 4);
    REQUIRE(engine->abort(batch->id));
    auto newcomer = seq::make(2, { 10 }, { .max_new_tokens = 1 });
    REQUIRE(newcomer);
    REQUIRE(engine->add(std::move(*newcomer)));
    batch = engine->schedule();
    REQUIRE(batch);
    REQUIRE(batch->items.size() == 2);
    CHECK(batch->items[0].token_count == 1);
    CHECK(engine->state().resources(1).blocks.size() == 2);
    commit(*engine, *batch);
    CHECK(engine->find_sequence(1)->processed_token_count() == 1);
    CHECK(engine->find_sequence(2)->is_finished());
    batch = engine->schedule();
    REQUIRE(batch);
    CHECK(batch->items[0].token_count == 4);
    commit(*engine, *batch);
    CHECK(engine->find_sequence(1)->processed_token_count() == 5);
}

TEST_CASE("a short newcomer and an existing stream advance beside a long prompt")
{
    auto config = test_config();
    config.kv_block_count = 32;
    config.max_batch_tokens = 4;
    config.prefill_chunk_tokens = 2;
    auto engine = scheduler::make(config, 99);
    REQUIRE(engine);
    auto stream = seq::make(1, { 10 }, { .max_new_tokens = 12 });
    REQUIRE(stream);
    REQUIRE(engine->add(std::move(*stream)));
    auto batch = engine->schedule();
    REQUIRE(batch);
    commit(*engine, *batch);
    auto long_prompt = seq::make(2, std::vector<token_id>(20, 10), { .max_new_tokens = 2 });
    auto short_prompt = seq::make(3, { 10, 20 }, { .max_new_tokens = 4 });
    REQUIRE(long_prompt);
    REQUIRE(short_prompt);
    REQUIRE(engine->add(std::move(*long_prompt)));
    REQUIRE(engine->add(std::move(*short_prompt)));
    for (int round = 0; round < 3; ++round) {
        batch = engine->schedule();
        REQUIRE(batch);
        CHECK(batch->phase == batch_phase::decode);
        commit(*engine, *batch);
        CHECK(engine->find_sequence(1)->completion_token_count() == std::size_t(round + 2));
        batch = engine->schedule();
        REQUIRE(batch);
        CHECK(batch->phase == batch_phase::prefill);
        commit(*engine, *batch);
        CHECK(engine->find_sequence(2)->processed_token_count() == std::size_t((round + 1) * 2));
        CHECK(engine->find_sequence(3)->completion_token_count() >= 1);
    }
}

TEST_CASE("small decode batches cover a whole round despite retries and cancellation")
{
    auto config = test_config();
    config.kv_block_count = 64;
    config.max_batch_tokens = 1;
    auto engine = scheduler::make(config, 99);
    REQUIRE(engine);
    for (const auto id : { 1, 2, 3 }) {
        auto sequence = seq::make(id, { 10 }, { .max_new_tokens = 10 });
        REQUIRE(sequence);
        REQUIRE(engine->add(std::move(*sequence)));
    }
    while (engine->waiting_count()) {
        auto batch = engine->schedule();
        REQUIRE(batch);
        commit(*engine, *batch);
    }
    auto newcomer = seq::make(4, { 20, 30 }, { .max_new_tokens = 2 });
    REQUIRE(newcomer);
    REQUIRE(engine->add(std::move(*newcomer)));
    auto batch = engine->schedule();
    REQUIRE(batch);
    CHECK(batch->phase == batch_phase::decode);
    CHECK(batch->items[0].id == 1);
    commit(*engine, *batch);

    SUBCASE("cancel an unvisited round member")
    {
        REQUIRE(engine->cancel(2));
    }
    SUBCASE("cancel a member already visited this round")
    {
        REQUIRE(engine->cancel(1));
        batch = engine->schedule();
        REQUIRE(batch);
        CHECK(batch->phase == batch_phase::decode);
        CHECK(batch->items[0].id == 2);
        commit(*engine, *batch);
    }
    batch = engine->schedule();
    REQUIRE(batch);
    CHECK(batch->phase == batch_phase::decode);
    CHECK(batch->items[0].id == 3);
    REQUIRE(engine->abort(batch->id));
    batch = engine->schedule();
    REQUIRE(batch);
    CHECK(batch->phase == batch_phase::decode);
    CHECK(batch->items[0].id == 3);
    commit(*engine, *batch);
    batch = engine->schedule();
    REQUIRE(batch);
    CHECK(batch->phase == batch_phase::prefill);
    CHECK(batch->items[0].id == 4);
}

TEST_CASE("admission commits future pages once and releases the budget at early EOS")
{
    auto config = test_config();
    config.kv_block_count = 4;
    auto engine = scheduler::make(config, 99);
    REQUIRE(engine);
    auto first = seq::make(1, { 10, 20, 30 }, { .max_new_tokens = 4 }); // 3 blocks
    auto second = seq::make(2, { 10 }, { .max_new_tokens = 2 });        // 1 block
    auto pending = seq::make(3, { 10 }, { .max_new_tokens = 1 });       // 1 block
    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(pending);
    CHECK(engine->try_add(*first).value() == chibillm::admission_result::admitted);
    CHECK(engine->try_add(*second).value() == chibillm::admission_result::admitted);
    CHECK(paged_state(*engine).used_block_count() == 2);
    CHECK(engine->try_add(*pending).value() == chibillm::admission_result::deferred);
    CHECK(std::ranges::equal(pending->tokens(), std::vector<token_id> { 10 }));
    CHECK(engine->find_sequence(3) == nullptr);
    auto batch = engine->schedule();
    REQUIRE(batch);
    REQUIRE(engine->complete(batch->id, std::array<token_id, 2> { 99, 42 }));
    CHECK(engine->find_sequence(1)->reason() == finish_reason::eos);
    // Finished sequences remain queryable but no longer commit any capacity.
    CHECK(engine->try_add(*pending).value() == chibillm::admission_result::admitted);
    REQUIRE(engine->cancel(2));
    REQUIRE(engine->cancel(3));
    CHECK(paged_state(*engine).used_block_count() == 0);
}

TEST_CASE("admission accounts for page boundaries and the unprocessed terminal sample")
{
    for (const auto [prompt, output] : { std::pair { 1u, 2u }, std::pair { 2u, 1u } }) {
        auto config = test_config();
        config.kv_block_count = 1;
        auto engine = scheduler::make(config, 99);
        REQUIRE(engine);
        auto sequence =
            seq::make(1, std::vector<token_id>(prompt, 10), { .max_new_tokens = output });
        REQUIRE(sequence);
        REQUIRE(engine->add(std::move(*sequence)));
        while (!engine->is_finished()) {
            auto batch = engine->schedule();
            REQUIRE(batch);
            CHECK(paged_state(*engine).used_block_count() == 1);
            commit(*engine, *batch);
        }
        CHECK(engine->find_sequence(1)->completion_token_count() == output);
        CHECK(paged_state(*engine).used_block_count() == 0);
    }
    auto engine = scheduler::make(test_config(), 99);
    REQUIRE(engine);
    auto overflow =
        seq::make(1, { 10, 20, 30 }, { .max_new_tokens = std::numeric_limits<std::size_t>::max() });
    REQUIRE(overflow);
    CHECK(engine->try_add(*overflow).value() == chibillm::admission_result::too_large);
}

TEST_CASE("sequence slots and cache commitments are reusable after cancellation")
{
    auto config = test_config();
    config.max_sequences = 1;
    auto engine = scheduler::make(config, 99);
    REQUIRE(engine);
    auto first = seq::make(1, { 10 }, { .max_new_tokens = 4 });
    auto next = seq::make(2, { 20 }, { .max_new_tokens = 4 });
    REQUIRE(first);
    REQUIRE(next);
    REQUIRE(engine->add(std::move(*first)));
    CHECK(engine->try_add(*next).value() == chibillm::admission_result::deferred);
    REQUIRE(engine->cancel(1));
    CHECK(engine->try_add(*next).value() == chibillm::admission_result::admitted);
    CHECK(engine->find_sequence(1)->is_finished());
    REQUIRE(engine->cancel(2));
}

TEST_CASE("failed state creation rolls back admission and leaves the request retryable")
{
    struct failing_state final : chibillm::model_state {
        chibillm::block_manager pages;
        bool fail_next { true };

        explicit failing_state(chibillm::block_manager pages)
            : pages(std::move(pages))
        {}

        std::size_t
        block_size() const noexcept override
        {
            return pages.block_size();
        }

        chibillm::sequence_resources
        resources(chibillm::seq_id id) const noexcept override
        {
            return pages.resources(id);
        }

        chibillm::result<void, chibillm::state_error>
        reserve(chibillm::seq_id id, std::size_t count) override
        {
            auto allocated = pages.reserve(id, count);
            if (!allocated)
                return allocated;
            if (std::exchange(fail_next, false))
                return chibillm::fail(chibillm::state_errc::allocation_failed);
            return {};
        }

        void
        release(chibillm::seq_id id) noexcept override
        {
            pages.release(id);
        }
    };

    auto pages = chibillm::block_manager::make(8, 2);
    REQUIRE(pages);
    auto storage = std::make_unique<failing_state>(std::move(*pages));
    const auto* state = storage.get();
    auto engine = scheduler::make(test_config(), 99, std::move(storage));
    REQUIRE(engine);
    auto sequence = seq::make(1, { 10, 20 }, { .max_new_tokens = 2 });
    REQUIRE(sequence);
    auto failed = engine->try_add(*sequence);
    REQUIRE_FALSE(failed);
    CHECK(failed.error() == scheduler_errc::block_manager_failure);
    CHECK(state->pages.used_block_count() == 0);
    CHECK(engine->find_sequence(1) == nullptr);
    CHECK(sequence->token_count() == 2);
    CHECK(engine->try_add(*sequence).value() == chibillm::admission_result::admitted);
    REQUIRE(engine->cancel(1));
    CHECK(state->pages.used_block_count() == 0);
}
