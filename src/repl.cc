#include "repl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "inference_engine.h"

namespace chibillm {
namespace {

enum class chat_errc {
    context_full,
    generation_failed
};

struct generation_result {
    std::string text;
    std::size_t prompt_tokens;
    std::size_t output_tokens;
    double time_to_first_token;
    double total_time;
    double decode_p50;
    double decode_p95;
    finish_reason reason;
    bool context_limited;
};

// No thread or per-token formatting. Only prefill uses a transient status line;
// once text arrives it owns the terminal until the final summary.
class generation_display {
public:
    generation_display(bool stream, bool progress)
        : stream_(stream)
        , progress_(progress)
        , next_refresh_(clock::now())
    {
        if (progress_)
            std::cerr << "[tokenizing]" << std::flush;
        else if (stream_)
            start_text();
    }

    ~generation_display()
    {
        clear();
        if (text_started_)
            std::cout << '\n' << std::flush;
    }

    void
    update(std::size_t processed, std::size_t total, std::chrono::steady_clock::time_point now)
    {
        if (!progress_ || text_started_ || now < next_refresh_)
            return;
        next_refresh_ = now + std::chrono::milliseconds(200);
        std::cerr
            << "\r\033[2K[prefill] "
            << processed
            << " / "
            << total
            << " tokens · "
            << (100 * processed / total)
            << "%"
            << std::flush;
    }

    void
    emit(const std::string& delta)
    {
        if (!stream_ || delta.empty())
            return;
        clear();
        start_text();
        std::cout << delta << std::flush;
    }

    void
    decoding()
    {
        clear();
        if (stream_)
            start_text();
    }

private:
    using clock = std::chrono::steady_clock;

    void
    clear()
    {
        if (progress_) {
            std::cerr << "\r\033[2K" << std::flush;
            progress_ = false;
        }
    }

    void
    start_text()
    {
        if (!text_started_) {
            std::cout << "qwen> " << std::flush;
            text_started_ = true;
        }
    }

    bool stream_;
    bool progress_;
    bool text_started_ = false;
    clock::time_point next_refresh_;
};

double
percentile(std::vector<double> samples, double fraction)
{
    if (samples.empty()) {
        return 0.0;
    }
    std::ranges::sort(samples);
    const auto index = std::min(samples.size() - 1,
                                static_cast<std::size_t>(std::ceil(fraction * samples.size()) - 1));
    return samples[index];
}

result<generation_result, chat_errc>
generate(chibillm::model_runner& runner,
         chibillm::inference_engine& engine,
         std::span<const chibillm::chat_message> history,
         std::size_t max_new_tokens,
         bool stream,
         bool progress)
{
    generation_display display(stream, progress);
    auto prompt = runner.encode_chat(history);
    if (!prompt) {
        return chibillm::fail(chat_errc::generation_failed);
    }
    const auto context_length = runner.info().max_context_tokens;
    if (prompt->size() >= context_length) {
        return chibillm::fail(chat_errc::context_full);
    }

    const auto prompt_tokens = prompt->size();
    const auto token_budget = std::min(max_new_tokens, context_length - prompt->size());
    auto sequence = chibillm::seq::make(1, std::move(*prompt),
                                        {
                                            .max_new_tokens = token_budget,
                                            .ignore_eos = false,
                                        });
    if (!sequence || !engine.add(std::move(*sequence))) {
        return chibillm::fail(chat_errc::generation_failed);
    }

    auto decoder = stream ? runner.make_decoder() : nullptr;
    if (stream && !decoder)
        return chibillm::fail(chat_errc::generation_failed);
    std::string text;
    using clock = std::chrono::steady_clock;
    const auto started = clock::now();
    auto first_token = started;
    bool produced_token = false;
    std::vector<double> decode_latencies;
    display.update(0, prompt_tokens, started);
    while (!engine.is_finished()) {
        const auto step_started = clock::now();
        auto updates = engine.step();
        if (!updates) {
            return chibillm::fail(chat_errc::generation_failed);
        }
        const auto step_finished = clock::now();
        if (produced_token) {
            decode_latencies.push_back(
                std::chrono::duration<double>(step_finished - step_started).count());
        }
        const auto* current = engine.find_sequence(1);
        if (!produced_token && current != nullptr && current->completion_token_count() != 0) {
            first_token = step_finished;
            produced_token = true;
            display.decoding();
        }
        if (!produced_token && current != nullptr)
            display.update(current->processed_token_count(), prompt_tokens, step_finished);
        if (stream) {
            for (const auto& update : *updates) {
                auto delta = decoder->push(update.token, update.reason != finish_reason::none);
                if (!delta)
                    return chibillm::fail(chat_errc::generation_failed);
                text += *delta;
                display.emit(*delta);
            }
        }
    }
    const auto finished_at = clock::now();

    const auto* finished = engine.find_sequence(1);
    if (finished == nullptr || !produced_token) {
        return chibillm::fail(chat_errc::generation_failed);
    }
    if (!stream) {
        auto response = runner.decode(finished->completion_tokens());
        if (!response)
            return chibillm::fail(chat_errc::generation_failed);
        text = std::move(*response);
    }
    return generation_result {
        .text = std::move(text),
        .prompt_tokens = prompt_tokens,
        .output_tokens = finished->completion_token_count(),
        .time_to_first_token = std::chrono::duration<double>(first_token - started).count(),
        .total_time = std::chrono::duration<double>(finished_at - started).count(),
        .decode_p50 = percentile(decode_latencies, 0.50),
        .decode_p95 = percentile(decode_latencies, 0.95),
        .reason = finished->reason(),
        .context_limited = token_budget < max_new_tokens,
    };
}

void
print_performance(const generation_result& result)
{
    const auto prefill_rate = result.prompt_tokens / result.time_to_first_token;
    const auto decode_tokens = result.output_tokens - 1;
    const auto decode_time = result.total_time - result.time_to_first_token;
    const auto decode_rate = decode_time > 0.0 ? decode_tokens / decode_time : 0.0;

    std::cerr
        << std::fixed
        << std::setprecision(2)
        << "[perf] prompt "
        << result.prompt_tokens
        << " tok | output "
        << result.output_tokens
        << " tok | first "
        << result.time_to_first_token
        << " s | prefill "
        << prefill_rate
        << " tok/s | decode "
        << decode_rate
        << " tok/s (p50 "
        << result.decode_p50 * 1000.0
        << " ms, p95 "
        << result.decode_p95 * 1000.0
        << " ms) | total "
        << result.total_time
        << " s\n";
    std::cerr
        << "[stop] "
        << (result.reason == finish_reason::eos ? "eos"
                : result.context_limited        ? "context capacity"
                                                : "token limit")
        << '\n';
}

} // namespace

int
run_repl(model_runner& runner,
         scheduler_config config,
         std::size_t max_new_tokens,
         bool stream,
         bool progress)
{
    // Both streams must be terminals: never put redraw controls in captured output.
    progress = progress && ::isatty(STDOUT_FILENO) && ::isatty(STDERR_FILENO);
    auto engine = inference_engine::make(config, runner);
    if (!engine) {
        std::cerr << "failed to create inference state\n";
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
        auto response = generate(runner, *engine, history, max_new_tokens, stream, progress);
        if (engine->find_sequence(1) && (!engine->cancel(1) || !engine->remove(1))) {
            std::cerr << "failed to release inference state\n";
            return 1;
        }
        if (!response) {
            history.pop_back();
            std::cerr << (response.error() == chat_errc::context_full
                              ? "conversation is too long; use /reset\n"
                              : "generation failed\n");
            continue;
        }

        if (!stream)
            std::cout << "qwen> " << response->text << '\n';
        print_performance(*response);
        history.push_back({ "assistant", std::move(response->text) });
    }
    return 0;
}

} // namespace chibillm
