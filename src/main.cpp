#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "inference_engine.h"
#include "qwen/qwen_model_runner.h"
#include "qwen/qwen_tokenizer.h"

namespace {

using chibillm::qwen_tokenizer;
using chibillm::result;

constexpr std::size_t kv_block_count = 64;
constexpr std::size_t kv_block_size = 16;
constexpr std::size_t max_context_tokens = kv_block_count * kv_block_size;
constexpr std::size_t max_new_tokens = 64;

struct chat_message {
    std::string role;
    std::string content;
};

result<std::vector<chibillm::token_id>, chibillm::qwen_tokenizer_errc>
format_chat_prompt(const qwen_tokenizer& tokenizer, std::span<const chat_message> messages)
{
    std::string prompt;
    for (const auto& message : messages) {
        prompt += "<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return tokenizer.encode(prompt);
}

enum class chat_errc {
    context_full,
    generation_failed
};

result<std::string, chat_errc>
generate(chibillm::qwen_model_runner& runner,
         const qwen_tokenizer& tokenizer,
         std::span<const chat_message> history)
{
    auto prompt = format_chat_prompt(tokenizer, history);
    if (!prompt) {
        return chibillm::fail(chat_errc::generation_failed);
    }
    if (prompt->size() >= max_context_tokens) {
        return chibillm::fail(chat_errc::context_full);
    }

    auto engine = chibillm::inference_engine::make(
        {
            .max_sequences = 1,
            .max_batch_tokens = 128,
            .kv_block_count = kv_block_count,
            .kv_block_size = kv_block_size,
            .eos_token = runner.config().eos_token_id,
        },
        runner);
    if (!engine) {
        return chibillm::fail(chat_errc::generation_failed);
    }

    const auto token_budget = std::min(max_new_tokens, max_context_tokens - prompt->size());
    auto sequence = chibillm::seq::make(1, std::move(*prompt),
                                        {
                                            .temperature = 1.0F,
                                            .max_new_tokens = token_budget,
                                            .ignore_eos = false,
                                        },
                                        kv_block_size);
    if (!sequence || !engine->add(std::move(*sequence))) {
        return chibillm::fail(chat_errc::generation_failed);
    }
    while (!engine->is_finished()) {
        if (!engine->step()) {
            return chibillm::fail(chat_errc::generation_failed);
        }
    }

    const auto* finished = engine->find_sequence(1);
    if (finished == nullptr) {
        return chibillm::fail(chat_errc::generation_failed);
    }
    auto response = tokenizer.decode(finished->completion_tokens());
    if (!response) {
        return chibillm::fail(chat_errc::generation_failed);
    }
    return std::move(*response);
}

std::string
load_text(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

} // namespace

int
main(int argc, char** argv)
{
    using namespace chibillm;

    const std::filesystem::path model_directory = argc > 1 ? argv[1] : "qwen_model";
    auto tokenizer = qwen_tokenizer::load(model_directory);
    if (!tokenizer) {
        std::cerr << "failed to load the tokenizer from " << model_directory << '\n';
        return 1;
    }
    const auto shader_source = load_text(CHIBILLM_SHADER_PATH);
    if (shader_source.empty()) {
        std::cerr << "failed to load Metal shaders\n";
        return 1;
    }
    auto runner =
        qwen_model_runner::make(model_directory, shader_source, kv_block_count, kv_block_size);
    if (!runner) {
        std::cerr << "failed to load Qwen from " << model_directory << '\n';
        return 1;
    }

    std::vector<chat_message> history;
    std::cout << "chibillm chat — /reset clears history, /quit exits\n";

    for (std::string input;;) {
        std::cout << "\nyou> " << std::flush;
        if (!std::getline(std::cin, input) || input == "/quit" || input == "/exit") {
            std::cout << '\n';
            break;
        }
        if (input == "/reset") {
            history.clear();
            std::cout << "history cleared\n";
            continue;
        }
        if (input.empty()) {
            continue;
        }

        history.push_back({ "user", input });
        auto response = generate(*runner, *tokenizer, history);
        if (!response) {
            history.pop_back();
            std::cerr << (response.error() == chat_errc::context_full
                              ? "conversation is too long; use /reset\n"
                              : "generation failed\n");
            continue;
        }

        std::cout << "qwen> " << *response << '\n';
        history.push_back({ "assistant", std::move(*response) });
    }
    return 0;
}
