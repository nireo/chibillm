#include "repl.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "generation_metrics.h"
#include "inference_engine.h"

namespace chibillm {
namespace {

using clock = std::chrono::steady_clock;

double
seconds(clock::time_point start, clock::time_point end)
{
    return std::chrono::duration<double>(end - start).count();
}

enum class chat_errc {
    context_full,
    generation_failed
};

// Only prefill uses a transient status line; generated text owns the terminal.
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
        finish();
    }

    void
    finish()
    {
        clear();
        if (text_started_) {
            std::cout << '\n' << std::flush;
            text_started_ = false;
        }
    }

    void
    update(std::size_t processed, std::size_t total, clock::time_point now)
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
        if (delta.empty())
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

    void
    buffered(const std::string& text)
    {
        clear();
        start_text();
        std::cout << text << std::flush;
    }

private:
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

result<std::string, chat_errc>
generate(model_runner& runner,
         inference_engine& engine,
         std::span<const chat_message> history,
         std::size_t max_new_tokens,
         bool stream,
         bool progress,
         clock::time_point submitted,
         generation_metrics& metrics)
{
    generation_display display(stream, progress);
    const auto encode_started = clock::now();
    auto prompt = runner.encode_chat(history);
    metrics.tokenization_seconds = seconds(encode_started, clock::now());
    if (!prompt)
        return fail(chat_errc::generation_failed);
    metrics.prompt_tokens = prompt->size();
    metrics.stage = "context_check";
    const auto context_length = runner.info().max_context_tokens;
    if (prompt->size() >= context_length) {
        metrics.stop = "context_capacity";
        return fail(chat_errc::context_full);
    }

    metrics.token_budget = std::min(max_new_tokens, context_length - prompt->size());
    metrics.stage = "admission";
    auto sequence = seq::make(1, std::move(*prompt),
                              { .max_new_tokens = metrics.token_budget, .ignore_eos = false });
    metrics.admission_at = seconds(submitted, clock::now());
    if (!sequence || !engine.add(std::move(*sequence)))
        return fail(chat_errc::generation_failed);

    metrics.stage = "decoder_setup";
    auto decoder = stream ? runner.make_decoder() : nullptr;
    if (stream && !decoder)
        return fail(chat_errc::generation_failed);
    std::string text;
    display.update(0, metrics.prompt_tokens, clock::now());
    while (!engine.is_finished()) {
        const auto* current = engine.find_sequence(1);
        if (!current)
            return fail(chat_errc::generation_failed);
        const auto processed_before = current->processed_token_count();
        const bool prefill = processed_before < metrics.prompt_tokens;
        metrics.stage = prefill ? "prefill" : "decode";
        const auto step_started = clock::now();
        auto updates = engine.step();
        const auto step_finished = clock::now();
        if (!updates)
            return fail(chat_errc::generation_failed);
        metrics.record_batch(prefill,
                             prefill ? current->processed_token_count() - processed_before : 0,
                             seconds(step_started, step_finished));
        if (prefill && !updates->empty())
            display.decoding();
        if (updates->empty())
            display.update(current->processed_token_count(), metrics.prompt_tokens, step_finished);
        for (const auto& update : *updates) {
            metrics.record_token(seconds(submitted, step_finished));
            if (stream) {
                metrics.stage = "text_decode";
                const auto decode_started = clock::now();
                auto delta = decoder->push(update.token, update.reason != finish_reason::none);
                if (delta) {
                    text += *delta;
                    metrics.output_bytes = text.size();
                }
                const auto decode_finished = clock::now();
                metrics.text_decode_seconds += seconds(decode_started, decode_finished);
                if (!delta)
                    return fail(chat_errc::generation_failed);
                if (!delta->empty()) {
                    metrics.stage = "output";
                    display.emit(*delta);
                    const auto written = clock::now();
                    metrics.output_write_seconds += seconds(decode_finished, written);
                    if (!std::cout)
                        return fail(chat_errc::generation_failed);
                    if (!metrics.first_text_seconds)
                        metrics.first_text_seconds = seconds(submitted, written);
                }
            }
        }
    }

    const auto* finished = engine.find_sequence(1);
    if (!finished || !metrics.first_token_at)
        return fail(chat_errc::generation_failed);
    if (!stream) {
        metrics.stage = "text_decode";
        const auto decode_started = clock::now();
        auto response = runner.decode(finished->completion_tokens());
        metrics.text_decode_seconds += seconds(decode_started, clock::now());
        if (!response)
            return fail(chat_errc::generation_failed);
        text = std::move(*response);
        metrics.output_bytes = text.size();
        metrics.stage = "output";
        const auto write_started = clock::now();
        display.buffered(text);
        const auto written = clock::now();
        metrics.output_write_seconds += seconds(write_started, written);
        if (!std::cout)
            return fail(chat_errc::generation_failed);
        if (!text.empty())
            metrics.first_text_seconds = seconds(submitted, written);
    }
    metrics.stage = "output";
    const auto write_started = clock::now();
    display.finish();
    metrics.output_write_seconds += seconds(write_started, clock::now());
    if (!std::cout)
        return fail(chat_errc::generation_failed);
    metrics.stop = finished->reason() == finish_reason::eos ? "eos"
        : metrics.token_budget < max_new_tokens             ? "context_capacity"
                                                            : "token_limit";
    metrics.status = "ok";
    metrics.stage = "complete";
    return text;
}

void
print_value(std::optional<double> value, double scale = 1)
{
    if (value)
        std::cerr << *value * scale;
    else
        std::cerr << "n/a";
}

void
print_performance(const generation_metrics& m)
{
    std::cerr
        << std::fixed
        << std::setprecision(3)
        << "[perf] prompt "
        << m.prompt_tokens
        << " tok | output "
        << m.output_tokens
        << " tok | tokenize "
        << m.tokenization_seconds * 1000
        << " ms | engine first ";
    print_value(m.engine_ttft_seconds());
    std::cerr << " s | first text ";
    print_value(m.first_text_seconds);
    std::cerr << " s | prefill " << m.prefill_seconds << " s (";
    print_value(m.prefill_tokens_per_second());
    std::cerr << " tok/s) | decode ";
    print_value(m.decode_tokens_per_second());
    std::cerr
        << " tok/s | total "
        << m.end_to_end_seconds
        << " s\n"
        << "[latency] decode step p50 ";
    print_value(m.decode_step.p50, 1000);
    std::cerr << " ms, p95 ";
    print_value(m.decode_step.p95, 1000);
    std::cerr << " ms (n=" << m.decode_step.samples << ") | inter-token p50 ";
    print_value(m.inter_token.p50, 1000);
    std::cerr << " ms, p95 ";
    print_value(m.inter_token.p95, 1000);
    std::cerr
        << " ms (n="
        << m.inter_token.samples
        << ") | text decode "
        << m.text_decode_seconds * 1000
        << " ms | output "
        << m.output_write_seconds * 1000
        << " ms\n[stop] "
        << (m.stop == "context_capacity"  ? "context capacity"
                : m.stop == "token_limit" ? "token limit"
                                          : m.stop)
        << '\n';
}

} // namespace

int
run_repl(model_runner& runner,
         scheduler_config config,
         std::size_t max_new_tokens,
         bool stream,
         bool progress,
         std::ostream* metrics_output)
{
    progress = progress && ::isatty(STDOUT_FILENO) && ::isatty(STDERR_FILENO);
    auto engine = inference_engine::make(config, runner);
    if (!engine) {
        std::cerr << "failed to create inference state\n";
        return 1;
    }
    // Disambiguates request IDs across append-mode sessions. Not a prompt identifier.
    const auto session_id = std::to_string(::getpid())
        + "-"
        + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    std::size_t request_id = 0;
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
        if (input.empty())
            continue;

        const auto submitted = clock::now();
        generation_metrics metrics;
        metrics.requested_tokens = max_new_tokens;
        ++request_id;
        history.push_back({ "user", input });
        auto response = generate(runner, *engine, history, max_new_tokens, stream, progress,
                                 submitted, metrics);
        // Ends after final answer flush (including buffered replies), before cleanup/reporting.
        metrics.end_to_end_seconds = seconds(submitted, clock::now());
        const bool cleanup_failed =
            engine->find_sequence(1) && (!engine->cancel(1) || !engine->remove(1));
        if (cleanup_failed) {
            metrics.status = "error";
            metrics.stage = "cleanup";
        }
        metrics.summarize();
        if (metrics_output
            && !write_metrics_jsonl(*metrics_output, metrics, runner.info(), config, session_id,
                                    request_id, stream, progress)) {
            std::cerr << "failed to write metrics\n";
            return 1;
        }
        if (cleanup_failed) {
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
        print_performance(metrics);
        history.push_back({ "assistant", std::move(*response) });
    }
    return 0;
}

} // namespace chibillm
