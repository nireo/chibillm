#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <limits>
#include <vector>

#include "seq.h"

using chibillm::finish_reason;
using chibillm::generation_params;
using chibillm::seq;
using chibillm::seq_errc;
using chibillm::seq_status;
using chibillm::token_id;

TEST_CASE("a sequence owns its prompt and starts waiting")
{
    const std::vector<token_id> prompt { 1, 2, 3 };
    auto result = seq::make(7, prompt, generation_params {});

    REQUIRE(result.has_value());
    const auto& sequence = *result;

    CHECK(sequence.id() == 7);
    CHECK(sequence.status() == seq_status::waiting);
    CHECK(sequence.reason() == finish_reason::none);
    CHECK(sequence.token_count() == 3);
    CHECK(sequence.prompt_token_count() == 3);
    CHECK(sequence.completion_token_count() == 0);
    CHECK(sequence.processed_token_count() == 0);
    CHECK(sequence.scheduled_token_count() == 0);
    CHECK(sequence.last_token() == 3);
    CHECK(sequence.prompt_tokens().size() == 3);
    CHECK(sequence.completion_tokens().empty());
}

TEST_CASE("invalid construction arguments return specific errors")
{
    CHECK(seq::make(1, {}, generation_params {}).error() == seq_errc::empty_prompt);
    CHECK(seq::make(1, { 1 }, generation_params { .max_new_tokens = 0 }).error()
          == seq_errc::invalid_max_new_tokens);
}

TEST_CASE("prefill and decode maintain the one-token cache gap")
{
    auto result = seq::make(1, { 10, 20, 30 }, generation_params {});
    REQUIRE(result.has_value());
    auto& sequence = *result;

    REQUIRE(sequence.schedule_tokens(3).has_value());
    CHECK(sequence.unprocessed_token_count() == 3);
    CHECK(sequence.schedulable_token_count() == 0);

    REQUIRE(sequence.commit_scheduled_tokens().has_value());
    REQUIRE(sequence.mark_running().has_value());
    REQUIRE(sequence.append_token(40).has_value());

    CHECK(sequence.token_count() == 4);
    CHECK(sequence.completion_token_count() == 1);
    CHECK(sequence.processed_token_count() == 3);
    CHECK(sequence.unprocessed_token_count() == 1);
    CHECK(sequence.completion_tokens().front() == 40);

    REQUIRE(sequence.schedule_tokens(1).has_value());
    REQUIRE(sequence.commit_scheduled_tokens().has_value());
    CHECK(sequence.processed_token_count() == 4);
}

TEST_CASE("a second scheduling reservation cannot overwrite the first")
{
    auto result = seq::make(1, { 1, 2, 3 }, generation_params {});
    REQUIRE(result.has_value());
    auto& sequence = *result;

    REQUIRE(sequence.schedule_tokens(2).has_value());

    auto second = sequence.schedule_tokens(1);
    REQUIRE_FALSE(second.has_value());
    CHECK(second.error() == seq_errc::work_already_scheduled);
    CHECK(sequence.scheduled_token_count() == 2);

    sequence.cancel_scheduled_tokens();
    CHECK(sequence.scheduled_token_count() == 0);
    CHECK(sequence.processed_token_count() == 0);
}

TEST_CASE("lifecycle transitions reject invalid source states")
{
    auto result = seq::make(1, { 1, 2 }, generation_params {});
    REQUIRE(result.has_value());
    auto& sequence = *result;

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
}

TEST_CASE("EOS and length limits are evaluated after appending")
{
    constexpr token_id eos = 99;

    auto eos_result = seq::make(1, { 1 }, generation_params {});
    REQUIRE(eos_result.has_value());
    auto& eos_sequence = *eos_result;

    CHECK(eos_sequence.evaluate_stop(eos) == finish_reason::none);
    REQUIRE(eos_sequence.schedule_tokens(1).has_value());
    REQUIRE(eos_sequence.commit_scheduled_tokens().has_value());
    REQUIRE(eos_sequence.mark_running().has_value());
    REQUIRE(eos_sequence.append_token(eos).has_value());
    CHECK(eos_sequence.evaluate_stop(eos) == finish_reason::eos);

    const generation_params ignored_eos {
        .max_new_tokens = 2,
        .ignore_eos = true,
    };
    auto limit_result = seq::make(2, { 1 }, ignored_eos);
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
    auto result = seq::make(1, { 1 }, generation_params {});
    REQUIRE(result.has_value());
    auto& sequence = *result;

    auto no_reason = sequence.finish(finish_reason::none);
    REQUIRE_FALSE(no_reason.has_value());
    CHECK(no_reason.error() == seq_errc::invalid_finish_reason);

    REQUIRE(sequence.finish(finish_reason::cancelled).has_value());
    CHECK(sequence.is_finished());
    CHECK(sequence.reason() == finish_reason::cancelled);

    auto scheduling = sequence.schedule_tokens(1);
    auto appending = sequence.append_token(2);

    REQUIRE_FALSE(scheduling.has_value());
    REQUIRE_FALSE(appending.has_value());
    CHECK(scheduling.error() == seq_errc::already_finished);
    CHECK(appending.error() == seq_errc::already_finished);
}
