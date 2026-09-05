#include "fake_model_runner.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <utility>
#include <vector>

#include "inference_engine.h"

using chibillm::fake_model_runner;
using chibillm::finish_reason;
using chibillm::generation_params;
using chibillm::inference_engine;
using chibillm::inference_engine_errc;
using chibillm::model_batch;
using chibillm::model_runner;
using chibillm::model_runner_errc;
using chibillm::result;
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

class stub_model_runner final : public model_runner {
public:
    explicit stub_model_runner(result<std::vector<token_id>, model_runner_errc> response)
        : response_(std::move(response))
    {}

    const chibillm::model_info&
    info() const noexcept override
    {
        static const chibillm::model_info value {
            .id = "test",
            .max_context_tokens = 16,
            .eos_token = 99,
        };
        return value;
    }

    result<std::vector<token_id>, model_runner_errc>
    encode_chat(std::span<const chibillm::chat_message>) override
    {
        return chibillm::fail(model_runner_errc::backend_failure);
    }

    result<std::string, model_runner_errc>
    decode(std::span<const token_id>) const override
    {
        return chibillm::fail(model_runner_errc::backend_failure);
    }

    result<std::vector<token_id>, model_runner_errc>
    execute(const model_batch&, chibillm::model_state&) override
    {
        return response_;
    }

private:
    result<std::vector<token_id>, model_runner_errc> response_;
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

    auto state = runner.make_state(test_config());
    REQUIRE(state.has_value());
    auto executed = runner.execute(empty, **state);
    REQUIRE_FALSE(executed.has_value());
    CHECK(executed.error() == model_runner_errc::empty_batch);
}

TEST_CASE("engine runs chunked prefill and decode until the length limit")
{
    fake_model_runner runner { 42 };
    auto engine = inference_engine::make(test_config(), runner);
    auto sequence = seq::make(1, { 10, 20, 30 },
                              generation_params {
                                  .max_new_tokens = 3,
                                  .ignore_eos = false,
                              });
    REQUIRE(engine.has_value());
    REQUIRE(sequence.has_value());
    REQUIRE(engine->add(std::move(*sequence)).has_value());

    std::size_t step_count = 0;
    std::size_t update_count = 0;
    while (!engine->is_finished()) {
        auto updates = engine->step();
        REQUIRE(updates.has_value());
        update_count += updates->size();
        ++step_count;
    }

    CHECK(step_count == 4);
    CHECK(update_count == 3);
    const auto* finished = engine->find_sequence(1);
    REQUIRE(finished != nullptr);
    CHECK(finished->status() == seq_status::finished);
    CHECK(finished->reason() == finish_reason::len_limit);
    REQUIRE(finished->completion_token_count() == 3);
    CHECK(finished->completion_tokens()[0] == 42);
    CHECK(finished->completion_tokens()[1] == 42);
    CHECK(finished->completion_tokens()[2] == 42);
}

TEST_CASE("runner failures abort the scheduler reservation")
{
    auto check_failure = [](result<std::vector<token_id>, model_runner_errc> response,
                            inference_engine_errc expected) {
        stub_model_runner runner(std::move(response));
        auto engine = inference_engine::make(test_config(), runner);
        auto sequence = seq::make(1, { 10 }, generation_params {});
        REQUIRE(engine.has_value());
        REQUIRE(sequence.has_value());
        REQUIRE(engine->add(std::move(*sequence)).has_value());

        auto stepped = engine->step();
        REQUIRE_FALSE(stepped.has_value());
        CHECK(stepped.error() == expected);
        CHECK_FALSE(engine->has_in_flight_batch());

        const auto* unchanged = engine->find_sequence(1);
        REQUIRE(unchanged != nullptr);
        CHECK(unchanged->processed_token_count() == 0);
        CHECK(unchanged->scheduled_token_count() == 0);
        CHECK(unchanged->status() == seq_status::waiting);
    };

    check_failure(chibillm::fail(model_runner_errc::backend_failure),
                  inference_engine_errc::model_execution_failed);
    check_failure(std::vector<token_id> {}, inference_engine_errc::runner_result_count_mismatch);
}

TEST_CASE(
    "non-paged model state rolls back failed execution and releases on completion or cancellation")
{
    struct recurrent_state final : chibillm::model_state {
        int value = 0;
        int snapshot = 0;
        int aborts = 0;
        int commits = 0;
        int releases = 0;

        result<void, chibillm::state_errc>
        reserve(chibillm::seq_id, std::size_t) override
        {
            return {};
        }

        void
        release(chibillm::seq_id) noexcept override
        {
            ++releases;
        }

        result<void, chibillm::state_errc>
        begin_batch(const model_batch&) override
        {
            snapshot = value;
            return {};
        }

        void
        abort_batch() noexcept override
        {
            value = snapshot;
            ++aborts;
        }

        void
        commit_batch() noexcept override
        {
            ++commits;
        }
    };

    class recurrent_runner final : public model_runner {
    public:
        mutable recurrent_state* state = nullptr;
        bool fail_next = true;

        const chibillm::model_info&
        info() const noexcept override
        {
            static const chibillm::model_info info { "recurrent-test", 16, 99 };
            return info;
        }

        result<std::unique_ptr<chibillm::model_state>, model_runner_errc>
        make_state(scheduler_config) const override
        {
            auto result = std::make_unique<recurrent_state>();
            state = result.get();
            return result;
        }

        result<std::vector<token_id>, model_runner_errc>
        encode_chat(std::span<const chibillm::chat_message>) override
        {
            return std::vector<token_id> { 1 };
        }

        result<std::string, model_runner_errc>
        decode(std::span<const token_id>) const override
        {
            return "x";
        }

        result<std::vector<token_id>, model_runner_errc>
        execute(const model_batch& batch, chibillm::model_state& storage) override
        {
            CHECK(batch.kv_block_size == 0);
            CHECK(batch.items[0].block_table.empty());
            auto& recurrent = dynamic_cast<recurrent_state&>(storage);
            recurrent.value += static_cast<int>(batch.tokens.size());
            if (std::exchange(fail_next, false))
                return chibillm::fail(model_runner_errc::backend_failure);
            return std::vector<token_id>(batch.sample_count(), 42);
        }
    } runner;

    auto config = test_config();
    config.kv_block_count = config.kv_block_size = 0;
    auto engine = inference_engine::make(config, runner);
    REQUIRE(engine.has_value());
    auto sequence = seq::make(1, { 1, 2, 3 }, { .max_new_tokens = 2 });
    REQUIRE(sequence.has_value());
    REQUIRE(engine->add(std::move(*sequence)).has_value());
    CHECK_FALSE(engine->step().has_value());
    CHECK(runner.state->value == 0);
    CHECK(runner.state->aborts == 1);
    auto intermediate = engine->step();
    REQUIRE(intermediate.has_value());
    CHECK(intermediate->empty());
    CHECK(runner.state->value == 2);
    CHECK(runner.state->commits == 1);
    auto first = engine->step();
    REQUIRE(first.has_value());
    REQUIRE(first->size() == 1);
    CHECK(first->front().token == 42);
    CHECK(runner.state->value == 3);
    SUBCASE("completion")
    {
        auto final = engine->step();
        REQUIRE(final.has_value());
        REQUIRE(final->size() == 1);
        CHECK(final->front().reason == finish_reason::len_limit);
        CHECK(runner.state->value == 4);
    }
    SUBCASE("cancellation")
    {
        REQUIRE(engine->cancel(1).has_value());
    }
    CHECK(runner.state->releases == 1);
    REQUIRE(engine->remove(1).has_value());
    CHECK(runner.state->releases == 1);
}

TEST_CASE("incremental decoding buffers UTF-8 tails and rejects invalid completed text")
{
    fake_model_runner runner(42);
    auto decoder = runner.make_decoder();
    REQUIRE(decoder->push(0xE2, false).value().empty());
    REQUIRE(decoder->push(0x82, false).value().empty());
    CHECK(decoder->push(0xAC, true).value() == "€");
    decoder = runner.make_decoder();
    CHECK_FALSE(decoder->push(0xE2, true).has_value());
    decoder = runner.make_decoder();
    REQUIRE(decoder->push(0xC0, false).has_value());
    CHECK_FALSE(decoder->push(0x80, true).has_value());
}
