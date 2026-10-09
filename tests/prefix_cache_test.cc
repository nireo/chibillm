#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "inference_engine.h"
#include "prefix_cache.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

using namespace chibillm;

namespace {
struct token_checkpoint final : model_checkpoint {
    explicit token_checkpoint(std::vector<token_id> data)
        : model_checkpoint(data.size())
        , tokens(std::move(data))
    {}

    std::vector<token_id> tokens;

    std::size_t
    size_bytes() const noexcept override
    {
        return tokens.size() * sizeof(token_id);
    }
};

// A toy recurrent backend: every sample depends on the entire processed prefix.
struct token_state final : model_state {
    std::unordered_map<seq_id, std::vector<token_id>> tokens;
    decltype(tokens) backup;
    bool reject_restore = false;
    bool reject_checkpoint = false;

    result<void, state_error>
    reserve(seq_id id, std::size_t) override
    {
        tokens.try_emplace(id);
        return {};
    }

    void
    release(seq_id id) noexcept override
    {
        tokens.erase(id);
    }

    std::size_t
    checkpoint_bytes(std::size_t count) const noexcept override
    {
        return count * sizeof(token_id);
    }

    result<std::unique_ptr<model_checkpoint>, state_error>
    checkpoint(seq_id id, std::size_t count) const override
    {
        if (reject_checkpoint || tokens.at(id).size() != count)
            return fail(state_errc::backend_failure);
        return std::make_unique<token_checkpoint>(tokens.at(id));
    }

    result<void, state_error>
    restore(seq_id id, const model_checkpoint& saved) override
    {
        if (std::exchange(reject_restore, false))
            return fail(state_errc::backend_failure);
        tokens.at(id) = dynamic_cast<const token_checkpoint&>(saved).tokens;
        return {};
    }

    result<void, state_error>
    begin_batch(const model_batch&) override
    {
        backup = tokens;
        return {};
    }

    void
    commit_batch() noexcept override
    {
        backup.clear();
    }

    void
    abort_batch() noexcept override
    {
        tokens = std::move(backup);
    }
};

struct checkpoint_runner final : model_runner {
    mutable token_state* storage = nullptr;
    bool fail_next = false;
    std::size_t prefill_tokens = 0;

    const model_info&
    info() const noexcept override
    {
        static const model_info value { "checkpoint-test", 256, -1 };
        return value;
    }

    result<std::unique_ptr<model_state>, model_runner_error>
    make_state(scheduler_config) const override
    {
        auto state = std::make_unique<token_state>();
        storage = state.get();
        return state;
    }

    result<encoded_prompt, model_runner_error>
    encode_chat(std::span<const chat_message>) override
    {
        return fail(model_runner_errc::invalid_chat);
    }

    result<std::string, model_runner_error>
    decode(std::span<const token_id>) const override
    {
        return std::string {};
    }

    result<std::vector<token_id>, model_runner_error>
    execute(const model_batch& batch, model_state&) override
    {
        if (batch.phase == batch_phase::prefill)
            prefill_tokens += batch.tokens.size();
        std::vector<token_id> samples;
        for (const auto& item : batch.items) {
            auto& history = storage->tokens.at(item.id);
            for (std::size_t i = item.token_offset; i < item.token_offset + item.token_count; ++i) {
                CHECK(batch.positions[i] == history.size());
                history.push_back(batch.tokens[i]);
            }
            if (item.logits_index)
                samples.push_back(1 + std::accumulate(history.begin(), history.end(), 0) % 10);
        }
        if (std::exchange(fail_next, false))
            return fail(model_runner_errc::backend_failure);
        return samples;
    }
};

scheduler_config
config()
{
    return { .max_sequences = 2,
             .max_batch_tokens = 16,
             .kv_block_count = 16,
             .kv_block_size = 2,
             .prefix_cache_bytes = 256,
             .prefix_cache_min_tokens = 1 };
}

std::vector<token_id>
finish(inference_engine& engine, seq_id id)
{
    while (!engine.is_finished())
        REQUIRE(engine.step());
    const auto* sequence = engine.find_sequence(id);
    REQUIRE(sequence);
    const auto tokens = sequence->completion_tokens();
    return { tokens.begin(), tokens.end() };
}

void
add(inference_engine& engine,
    seq_id id,
    std::vector<token_id> tokens,
    std::vector<std::size_t> boundaries = { 3 })
{
    auto sequence =
        seq::make(id, std::move(tokens), { .max_new_tokens = 3 }, std::move(boundaries));
    REQUIRE(sequence);
    REQUIRE(engine.add(std::move(*sequence)));
}
} // namespace

TEST_CASE("prefix LRU uses the longest exact proper prefix and bounds payload bytes")
{
    prefix_cache cache(32);
    const std::vector<token_id> a { 1, 2 }, b { 3, 4 }, c { 5, 6 };
    cache.insert(a, std::make_unique<token_checkpoint>(a));
    cache.insert(b, std::make_unique<token_checkpoint>(b));
    CHECK(cache.size_bytes() == 32);
    CHECK(cache.find(a) == nullptr);                        // Cached state does not include logits.
    REQUIRE(cache.find(std::vector<token_id> { 1, 2, 7 })); // Touch A.
    cache.insert(c, std::make_unique<token_checkpoint>(c));
    CHECK(cache.find(std::vector<token_id> { 3, 4, 7 }) == nullptr);
    REQUIRE(cache.find(std::vector<token_id> { 1, 2, 7 }));
    REQUIRE(cache.find(std::vector<token_id> { 5, 6, 7 }));
    CHECK_FALSE(cache.fits(1, 32));
    CHECK_FALSE(cache.fits(std::numeric_limits<std::size_t>::max(), 1));
    const std::vector<token_id> long_prefix { 1, 2, 3, 4, 5 };
    cache.insert(long_prefix, std::make_unique<token_checkpoint>(long_prefix));
    CHECK(cache.size_bytes() == 32);

    prefix_cache nested(64);
    const std::vector<token_id> short_prefix { 1 };
    nested.insert(a, std::make_unique<token_checkpoint>(a));
    nested.insert(short_prefix, std::make_unique<token_checkpoint>(short_prefix));
    const auto* best = nested.find(std::vector<token_id> { 1, 2, 8 });
    REQUIRE(best);
    CHECK(best->token_count == 2);
}

TEST_CASE("repeated and branching prompts restore private state and skip prefill")
{
    checkpoint_runner runner;
    auto engine = inference_engine::make(config(), runner);
    REQUIRE(engine);
    add(*engine, 1, { 1, 2, 3, 4, 5 });
    const auto expected = finish(*engine, 1);
    CHECK(runner.prefill_tokens == 5);
    REQUIRE(engine->remove(1));

    add(*engine, 1, { 1, 2, 3, 4, 5 });
    CHECK(engine->find_sequence(1)->cached_token_count() == 3);
    CHECK(finish(*engine, 1) == expected);
    CHECK(runner.prefill_tokens == 7);
    REQUIRE(engine->remove(1));

    add(*engine, 2, { 1, 2, 3, 8 });
    add(*engine, 3, { 1, 2, 3, 9 });
    CHECK(engine->find_sequence(2)->cached_token_count() == 3);
    CHECK(engine->find_sequence(3)->cached_token_count() == 3);
    const auto branch = finish(*engine, 2);
    const auto other = engine->find_sequence(3)->completion_tokens();
    const std::vector<token_id> other_tokens(other.begin(), other.end());

    checkpoint_runner cold_runner;
    auto cold_config = config();
    cold_config.prefix_cache_bytes = 0;
    auto cold = inference_engine::make(cold_config, cold_runner);
    REQUIRE(cold);
    add(*cold, 2, { 1, 2, 3, 8 });
    add(*cold, 3, { 1, 2, 3, 9 });
    CHECK(finish(*cold, 2) == branch);
    CHECK(std::ranges::equal(cold->find_sequence(3)->completion_tokens(), other_tokens));
}

TEST_CASE("failed batches publish no checkpoints and restored state remains retryable")
{
    checkpoint_runner runner;
    auto engine = inference_engine::make(config(), runner);
    REQUIRE(engine);
    add(*engine, 1, { 1, 2, 3, 4 });
    runner.fail_next = true;
    CHECK_FALSE(engine->step());
    CHECK(runner.storage->tokens.at(1).empty());
    REQUIRE(engine->cancel(1));
    REQUIRE(engine->remove(1));
    add(*engine, 2, { 1, 2, 3, 4 });
    CHECK(engine->find_sequence(2)->cached_token_count() == 0);
    const auto expected = finish(*engine, 2);
    REQUIRE(engine->remove(2));

    add(*engine, 3, { 1, 2, 3, 4 });
    CHECK(engine->find_sequence(3)->cached_token_count() == 3);
    runner.fail_next = true;
    CHECK_FALSE(engine->step());
    CHECK(runner.storage->tokens.at(3) == std::vector<token_id> { 1, 2, 3 });
    CHECK(finish(*engine, 3) == expected);
    REQUIRE(engine->remove(3));

    runner.storage->reject_restore = true;
    add(*engine, 4, { 1, 2, 3, 4 });
    CHECK(engine->find_sequence(4)->cached_token_count() == 0);
    CHECK(finish(*engine, 4) == expected);
}

TEST_CASE("disabled, short, oversized, and unavailable checkpoints do not split prefill")
{
    for (int mode = 0; mode < 4; ++mode) {
        auto settings = config();
        if (mode == 0)
            settings.prefix_cache_bytes = 0;
        if (mode == 1)
            settings.prefix_cache_min_tokens = 64;
        if (mode == 2)
            settings.prefix_cache_bytes = 1;
        checkpoint_runner runner;
        auto engine = inference_engine::make(settings, runner);
        REQUIRE(engine);
        runner.storage->reject_checkpoint = mode == 3;
        add(*engine, 1, { 1, 2, 3, 4 });
        if (mode < 3) {
            auto first = engine->step();
            REQUIRE(first);
            CHECK(first->size() == 1);
        }
        finish(*engine, 1);
        add(*engine, 2, { 1, 2, 3, 4 });
        CHECK(engine->find_sequence(2)->cached_token_count() == 0);
        finish(*engine, 2);
    }
}
