#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <utility>
#include <vector>

#include "inference_engine.h"

using chibillm::fake_model_runner;
using chibillm::finish_reason;
using chibillm::inference_engine;
using chibillm::inference_engine_errc;
using chibillm::model_batch;
using chibillm::model_runner;
using chibillm::model_runner_errc;
using chibillm::result;
using chibillm::sampling_params;
using chibillm::scheduler_config;
using chibillm::seq;
using chibillm::seq_status;
using chibillm::token_id;

namespace {

scheduler_config
test_config()
{
    return scheduler_config {
        .max_sequences = 4,
        .max_batch_tokens = 2,
        .kv_block_count = 8,
        .kv_block_size = 2,
        .eos_token = 99,
    };
}

class failing_model_runner final : public model_runner {
public:
    result<std::vector<token_id>, model_runner_errc>
    execute(const model_batch&) override
    {
        return chibillm::fail(model_runner_errc::backend_failure);
    }
};

class empty_result_model_runner final : public model_runner {
public:
    result<std::vector<token_id>, model_runner_errc>
    execute(const model_batch&) override
    {
        return std::vector<token_id> {};
    }
};

} // namespace

TEST_CASE("fake runner rejects malformed model batches")
{
    fake_model_runner runner { 42 };
    const model_batch empty {
        .id = 1,
        .phase = chibillm::batch_phase::prefill,
        .kv_block_size = 2,
        .tokens = {},
        .positions = {},
        .items = {},
    };

    auto executed = runner.execute(empty);
    REQUIRE_FALSE(executed.has_value());
    CHECK(executed.error() == model_runner_errc::empty_batch);
}

TEST_CASE("engine runs chunked prefill and decode until the length limit")
{
    fake_model_runner runner { 42 };
    auto engine = inference_engine::make(test_config(), runner);
    auto sequence = seq::make(1, { 10, 20, 30 },
                              sampling_params {
                                  .temperature = 1.0F,
                                  .max_new_tokens = 3,
                                  .ignore_eos = false,
                              },
                              2);
    REQUIRE(engine.has_value());
    REQUIRE(sequence.has_value());
    REQUIRE(engine->add(std::move(*sequence)).has_value());

    std::size_t step_count = 0;
    while (!engine->is_finished()) {
        REQUIRE(engine->step().has_value());
        ++step_count;
    }

    CHECK(step_count == 4);
    const auto* finished = engine->find_sequence(1);
    REQUIRE(finished != nullptr);
    CHECK(finished->status() == seq_status::finished);
    CHECK(finished->reason() == finish_reason::len_limit);
    REQUIRE(finished->completion_token_count() == 3);
    CHECK(finished->completion_tokens()[0] == 42);
    CHECK(finished->completion_tokens()[1] == 42);
    CHECK(finished->completion_tokens()[2] == 42);
}

TEST_CASE("model failure aborts the scheduler reservation")
{
    failing_model_runner runner;
    auto engine = inference_engine::make(test_config(), runner);
    auto sequence = seq::make(1, { 10 }, sampling_params {}, 2);
    REQUIRE(engine.has_value());
    REQUIRE(sequence.has_value());
    REQUIRE(engine->add(std::move(*sequence)).has_value());

    auto stepped = engine->step();
    REQUIRE_FALSE(stepped.has_value());
    CHECK(stepped.error() == inference_engine_errc::model_execution_failed);
    CHECK_FALSE(engine->has_in_flight_batch());

    const auto* unchanged = engine->find_sequence(1);
    REQUIRE(unchanged != nullptr);
    CHECK(unchanged->cached_token_count() == 0);
    CHECK(unchanged->scheduled_token_count() == 0);
    CHECK(unchanged->status() == seq_status::waiting);
}

TEST_CASE("wrong runner result count aborts the scheduler reservation")
{
    empty_result_model_runner runner;
    auto engine = inference_engine::make(test_config(), runner);
    auto sequence = seq::make(1, { 10 }, sampling_params {}, 2);
    REQUIRE(engine.has_value());
    REQUIRE(sequence.has_value());
    REQUIRE(engine->add(std::move(*sequence)).has_value());

    auto stepped = engine->step();
    REQUIRE_FALSE(stepped.has_value());
    CHECK(stepped.error() == inference_engine_errc::runner_result_count_mismatch);
    CHECK_FALSE(engine->has_in_flight_batch());

    const auto* unchanged = engine->find_sequence(1);
    REQUIRE(unchanged != nullptr);
    CHECK(unchanged->scheduled_token_count() == 0);
}
