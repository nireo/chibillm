#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "inference_engine.h"
#include "metal_test_support.h"
#include "model_factory.h"

#include <algorithm>
#include <filesystem>

using namespace chibillm;

TEST_CASE("Qwen cached repeat, shared system, continued chat and concurrent branches match "
          "uncached generation")
{
    const auto root = std::filesystem::path(QWEN3_5_MODEL_PATH).parent_path();
    for (const auto& directory : { root / "qwen_model", root / "qwen3_5_model" }) {
        if (!std::filesystem::exists(directory / "config.json")) {
            MESSAGE("Local Qwen model is not installed; skipping cache generation validation");
            continue;
        }
        CAPTURE(directory);
        auto loaded = load_model(directory, metal_test::kernel_source(), 32, 16, "cache-test");
        REQUIRE(loaded);
        auto& runner = **loaded;
        auto settings = scheduler_config {
            .max_sequences = 2, .max_batch_tokens = 32, .kv_block_count = 32, .kv_block_size = 16
        };
        auto warm = inference_engine::make(settings, runner);
        settings.prefix_cache_bytes = 0;
        auto cold = inference_engine::make(settings, runner);
        REQUIRE(warm);
        REQUIRE(cold);
        std::string system;
        for (int i = 0; i < 20; ++i)
            system += "Keep answers short. ";
        std::vector<chat_message> chat {
            { "system", system },
            { "user", "Reply with just the answer: What is 2 + 2?" },
        };
        auto first = runner.encode_chat(chat);
        REQUIRE(first);
        REQUIRE(first->checkpoints.size() == 2);
        REQUIRE(first->checkpoints.front() >= 64);

        const auto add = [&](inference_engine& engine, seq_id id, const encoded_prompt& prompt) {
            auto sequence =
                seq::make(id, prompt.tokens, { .max_new_tokens = 4 }, prompt.checkpoints);
            REQUIRE(sequence);
            REQUIRE(engine.add(std::move(*sequence)));
        };
        const auto finish = [&](inference_engine& engine, seq_id id) {
            while (!engine.is_finished())
                REQUIRE(engine.step());
            const auto tokens = engine.find_sequence(id)->completion_tokens();
            std::vector<token_id> result(tokens.begin(), tokens.end());
            REQUIRE(engine.remove(id));
            return result;
        };
        add(*cold, 1, *first);
        const auto expected = finish(*cold, 1);
        add(*warm, 1, *first);
        CHECK(warm->find_sequence(1)->cached_token_count() == 0);
        CHECK(finish(*warm, 1) == expected);
        add(*warm, 1, *first);
        CHECK(warm->find_sequence(1)->cached_token_count() == first->checkpoints.back());
        CHECK(finish(*warm, 1) == expected);

        chat.back().content = "Reply with just the answer: What is 3 + 3?";
        auto branch = runner.encode_chat(chat);
        REQUIRE(branch);
        add(*cold, 1, *branch);
        const auto branch_expected = finish(*cold, 1);
        add(*warm, 1, *branch);
        CHECK(warm->find_sequence(1)->cached_token_count() == first->checkpoints.front());
        CHECK(finish(*warm, 1) == branch_expected);

        chat.back().content = "Reply with just the answer: What is 2 + 2?";
        auto answer = runner.decode(expected);
        REQUIRE(answer);
        chat.push_back({ "assistant", *answer });
        chat.push_back({ "user", "Reply with just the answer: What is 5 + 5?" });
        auto continued = runner.encode_chat(chat);
        REQUIRE(continued);
        add(*cold, 1, *continued);
        const auto continued_expected = finish(*cold, 1);
        add(*warm, 1, *continued);
        CHECK(warm->find_sequence(1)->cached_token_count() == first->checkpoints.back());
        CHECK(finish(*warm, 1) == continued_expected);

        add(*warm, 2, *first);
        add(*warm, 3, *branch);
        CHECK(warm->find_sequence(2)->cached_token_count() == first->checkpoints.back());
        CHECK(warm->find_sequence(3)->cached_token_count() == branch->checkpoints.back());
        CHECK(finish(*warm, 2) == expected);
        CHECK(finish(*warm, 3) == branch_expected);
    }
}
