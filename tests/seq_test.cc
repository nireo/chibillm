#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <limits>
#include <vector>

#include "seq.h"

using chibillm::finish_reason;
using chibillm::sampling_params;
using chibillm::seq;
using chibillm::seq_errc;
using chibillm::seq_status;
using chibillm::token_id;

TEST_CASE("a sequence owns its prompt and starts waiting")
{
    const std::vector<token_id> prompt { 1, 2, 3 };
    auto result = seq::make(7, prompt, sampling_params {}, 16);

    REQUIRE(result.has_value());
    const auto& sequence = *result;

    CHECK(sequence.id() == 7);
    CHECK(sequence.status() == seq_status::waiting);
    CHECK(sequence.reason() == finish_reason::none);
    CHECK(sequence.token_count() == 3);
    CHECK(sequence.prompt_token_count() == 3);
    CHECK(sequence.completion_token_count() == 0);
    CHECK(sequence.cached_token_count() == 0);
    CHECK(sequence.scheduled_token_count() == 0);
    CHECK(sequence.last_token() == 3);
    CHECK(sequence.prompt_tokens().size() == 3);
    CHECK(sequence.completion_tokens().empty());
    CHECK(sequence.block_size() == 16);
}

TEST_CASE("invalid construction arguments return specific errors")
{
    auto empty = seq::make(1, {}, sampling_params {}, 16);
    REQUIRE_FALSE(empty.has_value());
    CHECK(empty.error() == seq_errc::empty_prompt);

    auto zero_temperature = seq::make(1, { 1 }, sampling_params { .temperature = 0.0f }, 16);
    REQUIRE_FALSE(zero_temperature.has_value());
    CHECK(zero_temperature.error() == seq_errc::invalid_temperature);

    auto nan_temperature = seq::make(
        1, { 1 }, sampling_params { .temperature = std::numeric_limits<float>::quiet_NaN() }, 16);
    REQUIRE_FALSE(nan_temperature.has_value());
    CHECK(nan_temperature.error() == seq_errc::invalid_temperature);

    auto infinite_temperature = seq::make(
        1, { 1 }, sampling_params { .temperature = std::numeric_limits<float>::infinity() }, 16);
    REQUIRE_FALSE(infinite_temperature.has_value());
    CHECK(infinite_temperature.error() == seq_errc::invalid_temperature);

    auto zero_output = seq::make(1, { 1 }, sampling_params { .max_new_tokens = 0 }, 16);
    REQUIRE_FALSE(zero_output.has_value());
    CHECK(zero_output.error() == seq_errc::invalid_max_new_tokens);

    auto zero_block = seq::make(1, { 1 }, sampling_params {}, 0);
    REQUIRE_FALSE(zero_block.has_value());
    CHECK(zero_block.error() == seq_errc::invalid_block_size);
}

TEST_CASE("prefill and decode maintain the one-token cache gap")
{
    auto result = seq::make(1, { 10, 20, 30 }, sampling_params {}, 2);
    REQUIRE(result.has_value());
    auto& sequence = *result;

    REQUIRE(sequence.schedule_tokens(3).has_value());
    CHECK(sequence.uncached_token_count() == 3);
    CHECK(sequence.schedulable_token_count() == 0);

    REQUIRE(sequence.commit_scheduled_tokens().has_value());
    REQUIRE(sequence.mark_running().has_value());
    REQUIRE(sequence.append_token(40).has_value());

    CHECK(sequence.token_count() == 4);
    CHECK(sequence.completion_token_count() == 1);
    CHECK(sequence.cached_token_count() == 3);
    CHECK(sequence.uncached_token_count() == 1);
    CHECK(sequence.completion_tokens().front() == 40);

    REQUIRE(sequence.schedule_tokens(1).has_value());
    REQUIRE(sequence.commit_scheduled_tokens().has_value());
    CHECK(sequence.cached_token_count() == 4);
}

TEST_CASE("a second scheduling reservation cannot overwrite the first")
{
    auto result = seq::make(1, { 1, 2, 3 }, sampling_params {}, 16);
    REQUIRE(result.has_value());
    auto& sequence = *result;

    REQUIRE(sequence.schedule_tokens(2).has_value());

    auto second = sequence.schedule_tokens(1);
    REQUIRE_FALSE(second.has_value());
    CHECK(second.error() == seq_errc::work_already_scheduled);
    CHECK(sequence.scheduled_token_count() == 2);

    sequence.cancel_scheduled_tokens();
    CHECK(sequence.scheduled_token_count() == 0);
    CHECK(sequence.cached_token_count() == 0);
}

TEST_CASE("lifecycle transitions reject invalid source states")
{
    auto result = seq::make(1, { 1, 2 }, sampling_params {}, 16);
    REQUIRE(result.has_value());
    auto& sequence = *result;

    CHECK(sequence.mark_waiting().error() == seq_errc::invalid_state_transition);
    CHECK(sequence.mark_running().error() == seq_errc::invalid_state_transition);
    CHECK(sequence.append_token(3).error() == seq_errc::invalid_state_transition);

    REQUIRE(sequence.schedule_tokens(2).has_value());
    CHECK(sequence.mark_running().error() == seq_errc::work_already_scheduled);

    REQUIRE(sequence.commit_scheduled_tokens().has_value());
    REQUIRE(sequence.mark_running().has_value());
    CHECK(sequence.mark_running().error() == seq_errc::invalid_state_transition);

    REQUIRE(sequence.append_token(3).has_value());
    CHECK(sequence.append_token(4).error() == seq_errc::invalid_state_transition);

    REQUIRE(sequence.schedule_tokens(1).has_value());
    REQUIRE(sequence.commit_scheduled_tokens().has_value());
    REQUIRE(sequence.mark_waiting().has_value());
    CHECK(sequence.status() == seq_status::waiting);
}

TEST_CASE("logical block geometry and slices handle exact and partial boundaries")
{
    auto result = seq::make(1, { 1, 2, 3, 4, 5, 6, 7 }, sampling_params {}, 3);
    REQUIRE(result.has_value());
    auto& sequence = *result;

    CHECK(sequence.logical_block_count() == 3);
    CHECK(sequence.tokens_in_last_block() == 1);

    auto first = sequence.logical_block_tokens(0);
    auto second = sequence.logical_block_tokens(1);
    auto third = sequence.logical_block_tokens(2);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(third.has_value());

    CHECK(first->size() == 3);
    CHECK((*first)[0] == 1);
    CHECK((*first)[2] == 3);
    CHECK(second->size() == 3);
    CHECK((*second)[0] == 4);
    CHECK((*second)[2] == 6);
    CHECK(third->size() == 1);
    CHECK((*third)[0] == 7);

    CHECK(sequence.logical_block_tokens(3).error() == seq_errc::index_out_of_range);

    REQUIRE(sequence.append_physical_block(11).has_value());
    REQUIRE(sequence.append_physical_block(22).has_value());
    REQUIRE(sequence.append_physical_block(33).has_value());
    CHECK(sequence.block_table().size() == 3);
    CHECK(sequence.block_table()[1] == 22);
    CHECK(sequence.append_physical_block(44).error() == seq_errc::block_table_full);

    auto exact = seq::make(2, { 1, 2, 3, 4 }, sampling_params {}, 4);
    REQUIRE(exact.has_value());
    CHECK(exact->logical_block_count() == 1);
    CHECK(exact->tokens_in_last_block() == 4);
}

TEST_CASE("cache metadata reset validates sequence state")
{
    auto result = seq::make(1, { 1, 2, 3, 4 }, sampling_params {}, 2);
    REQUIRE(result.has_value());
    auto& sequence = *result;

    REQUIRE(sequence.append_physical_block(10).has_value());
    REQUIRE(sequence.append_physical_block(20).has_value());
    REQUIRE(sequence.set_cached_token_count(2).has_value());
    REQUIRE(sequence.schedule_tokens(2).has_value());

    CHECK(sequence.reset_cache_metadata().error() == seq_errc::work_already_scheduled);
    CHECK(sequence.cached_token_count() == 2);
    CHECK(sequence.block_table().size() == 2);

    sequence.cancel_scheduled_tokens();
    REQUIRE(sequence.reset_cache_metadata().has_value());
    CHECK(sequence.cached_token_count() == 0);
    CHECK(sequence.block_table().empty());
    CHECK(sequence.token_count() == 4);

    REQUIRE(sequence.schedule_tokens(4).has_value());
    REQUIRE(sequence.commit_scheduled_tokens().has_value());
    REQUIRE(sequence.mark_running().has_value());
    CHECK(sequence.reset_cache_metadata().error() == seq_errc::invalid_state_transition);
}

TEST_CASE("EOS and length limits are evaluated after appending")
{
    constexpr token_id eos = 99;

    auto eos_result = seq::make(1, { 1 }, sampling_params {}, 16);
    REQUIRE(eos_result.has_value());
    auto& eos_sequence = *eos_result;

    CHECK(eos_sequence.evaluate_stop(eos) == finish_reason::none);
    REQUIRE(eos_sequence.schedule_tokens(1).has_value());
    REQUIRE(eos_sequence.commit_scheduled_tokens().has_value());
    REQUIRE(eos_sequence.mark_running().has_value());
    REQUIRE(eos_sequence.append_token(eos).has_value());
    CHECK(eos_sequence.evaluate_stop(eos) == finish_reason::eos);

    const sampling_params ignored_eos {
        .temperature = 1.0f,
        .max_new_tokens = 2,
        .ignore_eos = true,
    };
    auto limit_result = seq::make(2, { 1 }, ignored_eos, 16);
    REQUIRE(limit_result.has_value());
    auto& limit_sequence = *limit_result;

    REQUIRE(limit_sequence.schedule_tokens(1).has_value());
    REQUIRE(limit_sequence.commit_scheduled_tokens().has_value());
    REQUIRE(limit_sequence.mark_running().has_value());
    REQUIRE(limit_sequence.append_token(eos).has_value());
    CHECK(limit_sequence.evaluate_stop(eos) == finish_reason::none);

    REQUIRE(limit_sequence.schedule_tokens(1).has_value());
    REQUIRE(limit_sequence.commit_scheduled_tokens().has_value());
    REQUIRE(limit_sequence.append_token(5).has_value());
    CHECK(limit_sequence.evaluate_stop(eos) == finish_reason::len_limit);
}

TEST_CASE("finished sequences cannot be revived or mutated")
{
    auto result = seq::make(1, { 1 }, sampling_params {}, 16);
    REQUIRE(result.has_value());
    auto& sequence = *result;

    auto no_reason = sequence.finish(finish_reason::none);
    REQUIRE_FALSE(no_reason.has_value());
    CHECK(no_reason.error() == seq_errc::invalid_finish_reason);

    REQUIRE(sequence.finish(finish_reason::cancelled).has_value());
    CHECK(sequence.is_finished());
    CHECK(sequence.reason() == finish_reason::cancelled);

    auto waiting = sequence.mark_waiting();
    auto scheduling = sequence.schedule_tokens(1);
    auto appending = sequence.append_token(2);

    REQUIRE_FALSE(waiting.has_value());
    REQUIRE_FALSE(scheduling.has_value());
    REQUIRE_FALSE(appending.has_value());
    CHECK(waiting.error() == seq_errc::already_finished);
    CHECK(scheduling.error() == seq_errc::already_finished);
    CHECK(appending.error() == seq_errc::already_finished);
}
