#include <doctest/doctest.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>

#include "serving_runtime.h"

namespace {
using namespace chibillm;
using namespace std::chrono_literals;

// Tests grant one model step at a time; no sleeps or timing-dependent arrival order.
class controlled_runner final : public model_runner {
public:
    const model_info&
    info() const noexcept override
    {
        static const model_info info { "controlled", 64, 99 };
        return info;
    }

    result<std::vector<token_id>, model_runner_error>
    encode_chat(std::span<const chat_message> messages) override
    {
        return std::vector<token_id>(messages.front().content.size(), 10);
    }

    result<std::string, model_runner_error>
    decode(std::span<const token_id> tokens) const override
    {
        return std::string(tokens.size(), '*');
    }

    result<std::vector<token_id>, model_runner_error>
    execute(const model_batch& batch, model_state&) override
    {
        std::unique_lock lock(mutex_);
        batches_.push_back(batch);
        changed_.notify_all();
        changed_.wait(lock, [&] { return released_ || permits_; });
        if (!released_)
            --permits_;
        return std::vector<token_id>(batch.sample_count(), 42);
    }

    std::optional<model_batch>
    wait_batch(std::size_t count)
    {
        std::unique_lock lock(mutex_);
        if (!changed_.wait_for(lock, 2s, [&] { return batches_.size() >= count; }))
            return std::nullopt;
        return batches_[count - 1];
    }

    void
    advance()
    {
        std::lock_guard lock(mutex_);
        ++permits_;
        changed_.notify_all();
    }

    void
    release()
    {
        std::lock_guard lock(mutex_);
        released_ = true;
        changed_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<model_batch> batches_;
    std::size_t permits_ {};
    bool released_ {};
};

struct runtime_session {
    controlled_runner runner;
    std::unique_ptr<serving_runtime> runtime;

    ~runtime_session()
    {
        runner.release();
    } // Unblock execution before runtime destruction joins.
};

generation_request
request(std::size_t prompt, std::size_t output)
{
    return { { { "user", std::string(prompt, 'a') } }, output };
}

std::shared_ptr<request_state>
prepare_next(runtime_session& session, generation_request request)
{
    auto submitted = session.runtime->enqueue(std::move(request));
    REQUIRE(submitted);
    session.runner.advance();
    const auto& state = *submitted;
    std::unique_lock lock(state->mutex);
    REQUIRE(state->changed.wait_for(lock, 2s,
                                    [&] { return state->status != request_status::submitted; }));
    REQUIRE_FALSE(state->error);
    return state;
}

void
wait_finished(const std::shared_ptr<request_state>& state)
{
    std::unique_lock lock(state->mutex);
    REQUIRE(state->changed.wait_for(lock, 2s,
                                    [&] { return state->status == request_status::finished; }));
    CHECK_FALSE(state->error);
}
} // namespace

TEST_CASE("FIFO cache admission waits for completion or cancellation without bypass")
{
    runtime_session session;
    auto runtime = serving_runtime::make(session.runner,
                                         { .scheduler = { .max_sequences = 4,
                                                          .max_batch_tokens = 2,
                                                          .kv_block_count = 4,
                                                          .kv_block_size = 2,
                                                          .prefill_chunk_tokens = 1 } });
    REQUIRE(runtime);
    session.runtime = std::move(*runtime);

    auto first = session.runtime->submit(request(3, 4)); // Commits 3 of the 4 blocks.
    REQUIRE(first);
    REQUIRE(session.runner.wait_batch(1));
    auto second = prepare_next(session, request(3, 2)); // Needs 2 blocks: must wait.
    auto batch = session.runner.wait_batch(2);
    REQUIRE(batch);
    REQUIRE(batch->items.size() == 1);
    CHECK(batch->items[0].id == (*first)->id);
    auto third = prepare_next(session, request(1, 1)); // Fits, but cannot bypass the second.
    batch = session.runner.wait_batch(3);
    REQUIRE(batch);
    REQUIRE(batch->items.size() == 1);
    CHECK(batch->items[0].id == (*first)->id);
    CHECK(batch->phase == batch_phase::decode);

    SUBCASE("early cancellation releases the FIFO head")
    {
        session.runtime->cancel(second);
        session.runner.advance();
        batch = session.runner.wait_batch(4);
        REQUIRE(batch);
        REQUIRE(batch->items.size() == 1);
        CHECK(batch->items[0].id == third->id);
        CHECK(batch->phase == batch_phase::prefill);
    }
    SUBCASE("completion releases the committed budget")
    {
        session.runner.advance();
        batch = session.runner.wait_batch(4);
        REQUIRE(batch);
        CHECK(batch->items[0].id == (*first)->id);
        session.runner.advance();
        batch = session.runner.wait_batch(5);
        REQUIRE(batch);
        CHECK(batch->items[0].id == (*first)->id);
        session.runner.advance();
        batch = session.runner.wait_batch(6);
        REQUIRE(batch);
        CHECK(batch->items[0].id == second->id);
    }

    session.runner.release();
    wait_finished(*first);
    wait_finished(second);
    wait_finished(third);
    CHECK((*first)->output == "****");
    CHECK(third->output == "*");
    CHECK((second->reason == finish_reason::cancelled || second->output == "**"));
    auto next = session.runtime->submit(request(3, 4));
    REQUIRE(next);
    wait_finished(*next);
    CHECK((*next)->output == "****");
}

TEST_CASE("oversized requests and per-request context limits fail before model execution")
{
    runtime_session session;
    auto runtime = serving_runtime::make(
        session.runner, { .scheduler = { .kv_block_count = 2, .kv_block_size = 2 } });
    REQUIRE(runtime);
    session.runtime = std::move(*runtime);
    auto oversized = session.runtime->submit(request(4, 2));
    REQUIRE_FALSE(oversized);
    CHECK(oversized.error().kind == generation_errc::invalid_input);
    CHECK(oversized.error().code == "cache_capacity_exceeded");
    session.runtime.reset();
    runtime = serving_runtime::make(
        session.runner,
        { .scheduler = { .kv_block_count = 8, .kv_block_size = 2 }, .max_context_tokens = 4 });
    REQUIRE(runtime);
    session.runtime = std::move(*runtime);
    auto beyond_context = session.runtime->submit(request(3, 2));
    REQUIRE_FALSE(beyond_context);
    CHECK(beyond_context.error().code == "context_length_exceeded");
    session.runner.release();
    auto valid = session.runtime->submit(request(2, 2));
    REQUIRE(valid);
    wait_finished(*valid);
    CHECK((*valid)->output == "**");
}
