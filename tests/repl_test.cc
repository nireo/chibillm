#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <functional>
#include <iostream>
#include <sstream>
#include <utility>

#include "cli_options.h"
#include "model_runner.h"
#include "repl.h"
#include "scheduler.h"

namespace {

struct captured_terminal {
    std::istringstream input;
    std::ostringstream output;
    std::ostringstream errors;
    std::streambuf* old_input;
    std::streambuf* old_output;
    std::streambuf* old_errors;
    std::ios::fmtflags error_flags;
    std::streamsize error_precision;

    explicit captured_terminal(std::string text)
        : input(std::move(text))
        , old_input(std::cin.rdbuf(input.rdbuf()))
        , old_output(std::cout.rdbuf(output.rdbuf()))
        , old_errors(std::cerr.rdbuf(errors.rdbuf()))
        , error_flags(std::cerr.flags())
        , error_precision(std::cerr.precision())
    {}

    ~captured_terminal()
    {
        std::cin.rdbuf(old_input);
        std::cout.rdbuf(old_output);
        std::cerr.rdbuf(old_errors);
        std::cerr.flags(error_flags);
        std::cerr.precision(error_precision);
        std::cin.clear();
    }
};

class repl_runner final : public chibillm::model_runner {
public:
    chibillm::model_info metadata { "test", 16, 99 };
    std::size_t calls = 0;
    std::size_t prompt_size = 5;
    std::size_t failure_call = 0;
    bool bad_decode = false;
    bool immediate_eos = false;
    std::vector<std::vector<chibillm::chat_message>> histories;
    std::function<void(std::size_t)> before_execute;

    const chibillm::model_info&
    info() const noexcept override
    {
        return metadata;
    }

    chibillm::result<std::vector<chibillm::token_id>, chibillm::model_runner_errc>
    encode_chat(std::span<const chibillm::chat_message> messages) override
    {
        histories.emplace_back(messages.begin(), messages.end());
        calls = 0;
        return std::vector<chibillm::token_id>(prompt_size, 1);
    }

    chibillm::result<std::string, chibillm::model_runner_errc>
    decode(std::span<const chibillm::token_id> tokens) const override
    {
        if (bad_decode)
            return chibillm::fail(chibillm::model_runner_errc::tokenizer_failure);
        std::string text;
        for (auto token : tokens) {
            if (token == 10)
                text += "\xe2";
            if (token == 11)
                text += "\x82\xac";
        }
        return text;
    }

    chibillm::result<std::vector<chibillm::token_id>, chibillm::model_runner_errc>
    execute(const chibillm::model_batch& batch, chibillm::model_state&) override
    {
        ++calls;
        if (before_execute)
            before_execute(calls);
        if (calls == failure_call)
            return chibillm::fail(chibillm::model_runner_errc::backend_failure);
        const auto prefill_calls = (prompt_size + 1) / 2;
        const chibillm::token_id token = immediate_eos ? 99
            : calls == prefill_calls                   ? 10
            : calls == prefill_calls + 1               ? 11
                                                       : 99;
        return std::vector<chibillm::token_id>(batch.sample_count(), token);
    }
};

int
run(repl_runner& runner,
    bool stream = true,
    std::size_t budget = 8,
    std::ostream* metrics = nullptr)
{
    return chibillm::run_repl(runner,
                              { .max_sequences = 1,
                                .max_batch_tokens = 2,
                                .kv_block_count = 8,
                                .kv_block_size = 2,
                                .eos_token = 99 },
                              budget, stream, false, metrics);
}

} // namespace

TEST_CASE("REPL streams complete UTF-8 before generation finishes")
{
    captured_terminal terminal("hello\n/quit\n");
    repl_runner runner;
    bool prefix_before_execution = false;
    bool withheld_partial = false;
    bool emitted_before_eos = false;
    runner.before_execute = [&](std::size_t call) {
        if (call == 1)
            prefix_before_execution = terminal.output.str().find("qwen> ") != std::string::npos;
        if (call == 4)
            withheld_partial = terminal.output.str().ends_with("qwen> ");
        if (call == 5)
            emitted_before_eos = terminal.output.str().find("€") != std::string::npos;
    };
    const auto status = run(runner);
    CHECK(status == 0);
    CHECK(prefix_before_execution);
    CHECK(withheld_partial);
    CHECK(emitted_before_eos);
    CHECK(terminal.output.str().find("qwen> €\n") != std::string::npos);
    CHECK(terminal.output.str().find('\033') == std::string::npos);
    CHECK(terminal.errors.str().find('\033') == std::string::npos);
    CHECK(terminal.errors.str().find("[stop] eos") != std::string::npos);
}

TEST_CASE("streamed and buffered REPL output and history agree")
{
    std::string streamed;
    std::string buffered;
    for (bool stream : { true, false }) {
        captured_terminal terminal("hello\nagain\n/quit\n");
        repl_runner runner;
        const auto status = run(runner, stream);
        CHECK(status == 0);
        REQUIRE(runner.histories.size() == 2);
        REQUIRE(runner.histories[1].size() == 3);
        CHECK(runner.histories[1][1].role == "assistant");
        CHECK(runner.histories[1][1].content == "€");
        (stream ? streamed : buffered) = terminal.output.str();
    }
    CHECK(streamed == buffered);
}

TEST_CASE("partial streaming failure releases state and rolls back history")
{
    captured_terminal terminal("hello\nretry\n/quit\n");
    repl_runner runner;
    runner.failure_call = 5;
    const auto status = run(runner);
    CHECK(status == 0);
    CHECK(terminal.output.str().find("qwen> €\n") != std::string::npos);
    CHECK(terminal.errors.str().find("generation failed") != std::string::npos);
    REQUIRE(runner.histories.size() == 2);
    REQUIRE(runner.histories[1].size() == 1);
    CHECK(runner.histories[1][0].content == "retry");
}

TEST_CASE("REPL reports output budget and context limits")
{
    for (bool context_limited : { false, true }) {
        captured_terminal terminal("hello\n/quit\n");
        repl_runner runner;
        if (context_limited)
            runner.metadata.max_context_tokens = 7;
        const auto status = run(runner, true, context_limited ? 8 : 2);
        CHECK(status == 0);
        CHECK(terminal.output.str().find("qwen> €\n") != std::string::npos);
        CHECK(terminal.errors.str().find(context_limited ? "[stop] context capacity"
                                                         : "[stop] token limit")
              != std::string::npos);
    }
}

TEST_CASE("decoder errors and full contexts leave the REPL usable")
{
    for (bool context_full : { false, true }) {
        captured_terminal terminal("hello\nretry\n/quit\n");
        repl_runner runner;
        runner.bad_decode = !context_full;
        if (context_full)
            runner.metadata.max_context_tokens = 5;
        const auto status = run(runner);
        CHECK(status == 0);
        CHECK(terminal.errors.str().find(context_full ? "conversation is too long"
                                                      : "generation failed")
              != std::string::npos);
        REQUIRE(runner.histories.size() == 2);
        CHECK(runner.histories[1].size() == 1);
    }
}

TEST_CASE("immediate EOS finishes an empty streamed response")
{
    captured_terminal terminal("hello\n/quit\n");
    repl_runner runner;
    runner.immediate_eos = true;
    const auto status = run(runner);
    CHECK(status == 0);
    CHECK(terminal.output.str().find("qwen> \n") != std::string::npos);
    CHECK(terminal.errors.str().find("[stop] eos") != std::string::npos);
    CHECK(terminal.errors.str().find("generation failed") == std::string::npos);
}

TEST_CASE("CLI defaults to Qwen3.5 and accepts an explicit model directory")
{
    char program[] = "chibillm";
    char directory[] = "qwen_model";
    char* args[] = { program, directory };
    auto settings = chibillm::parse_cli_options(1, args);
    REQUIRE(settings.has_value());
    CHECK(settings->model_directory == "qwen3_5_model");
    settings = chibillm::parse_cli_options(2, args);
    REQUIRE(settings.has_value());
    CHECK(settings->model_directory == "qwen_model");
}

TEST_CASE("CLI accepts efficient REPL display controls")
{
    char program[] = "chibillm";
    char no_stream[] = "--no-stream";
    char off[] = "--progress=off";
    char automatic[] = "--progress=auto";
    char invalid[] = "--progress=always";
    char* args[] = { program, no_stream, off, automatic };
    auto settings = chibillm::parse_cli_options(3, args);
    REQUIRE(settings.has_value());
    CHECK_FALSE(settings->stream);
    CHECK_FALSE(settings->progress);
    settings = chibillm::parse_cli_options(4, args);
    REQUIRE(settings.has_value());
    CHECK(settings->progress);
    args[1] = invalid;
    CHECK_FALSE(chibillm::parse_cli_options(2, args).has_value());
}

TEST_CASE("REPL metrics count committed batches and separate first token from text")
{
    for (bool stream : { true, false }) {
        captured_terminal terminal("private prompt\n/quit\n");
        std::ostringstream records;
        repl_runner runner;
        CHECK(run(runner, stream, 8, &records) == 0);
        const auto row = nlohmann::json::parse(records.str());
        CHECK(row["status"] == "ok");
        CHECK(row["stage"] == "complete");
        CHECK(row["stop_reason"] == "eos");
        CHECK(row["prompt_tokens"] == 5);
        CHECK(row["processed_prompt_tokens"] == 5);
        CHECK(row["prefill_batches"] == 3);
        CHECK(row["output_tokens"] == 3);
        CHECK(row["output_bytes"] == 3);
        CHECK(row["decode_step"]["samples"] == 2);
        CHECK(row["inter_token"]["samples"] == 2);
        CHECK(row["stream"] == stream);
        CHECK(row["first_text_seconds"].get<double>() >= row["engine_ttft_seconds"].get<double>());
        CHECK(row["end_to_end_seconds"].get<double>() >= row["first_text_seconds"].get<double>());
        CHECK(row["prefill_seconds"].get<double>() + row["decode_engine_seconds"].get<double>()
              <= row["end_to_end_seconds"].get<double>());
        CHECK(records.str().find("private prompt") == std::string::npos);
        CHECK(records.str().find("€") == std::string::npos);
    }
}

TEST_CASE("failed requests produce partial metrics and distinct request IDs")
{
    captured_terminal terminal("hello\n/reset\nretry\n/quit\n");
    std::ostringstream records;
    repl_runner runner;
    runner.failure_call = 5;
    CHECK(run(runner, true, 8, &records) == 0);
    std::istringstream lines(records.str());
    std::string line;
    for (int request = 1; request <= 2; ++request) {
        REQUIRE(static_cast<bool>(std::getline(lines, line)));
        const auto row = nlohmann::json::parse(line);
        CHECK(row["request_id"] == request);
        CHECK(row["status"] == "error");
        CHECK(row["stage"] == "decode");
        CHECK(row["output_tokens"] == 2);
        CHECK(row["output_bytes"] == 3);
        CHECK(row["decode_step"]["samples"] == 1); // Failed batch isn't committed work.
    }
    CHECK_FALSE(static_cast<bool>(std::getline(lines, line)));
}

TEST_CASE("metrics distinguish no text, early rejection and file write errors")
{
    for (bool reject : { true, false }) {
        captured_terminal terminal("hello\n/quit\n");
        std::ostringstream records;
        repl_runner runner;
        runner.immediate_eos = true;
        if (reject)
            runner.metadata.max_context_tokens = 5;
        CHECK(run(runner, true, 8, &records) == 0);
        const auto row = nlohmann::json::parse(records.str());
        CHECK(row["first_text_seconds"].is_null());
        CHECK(row["decode_step"]["p50_seconds"].is_null());
        CHECK(row["engine_ttft_seconds"].is_null() == reject);
        CHECK(row["status"] == (reject ? "error" : "ok"));
        CHECK(row["stage"] == (reject ? "context_check" : "complete"));
    }
    captured_terminal terminal("hello\n/quit\n");
    std::ostringstream failed;
    failed.setstate(std::ios::badbit);
    repl_runner runner;
    CHECK(run(runner, true, 8, &failed) == 1);
    CHECK(terminal.errors.str().find("failed to write metrics") != std::string::npos);
}

TEST_CASE("CLI validates REPL-only JSONL output")
{
    char program[] = "chibillm";
    char option[] = "--metrics-jsonl";
    char path[] = "metrics.jsonl";
    char serve[] = "--serve";
    char* args[] = { program, option, path, serve };
    CHECK_FALSE(chibillm::parse_cli_options(2, args).has_value());
    auto settings = chibillm::parse_cli_options(3, args);
    REQUIRE(settings.has_value());
    CHECK(settings->metrics_jsonl == "metrics.jsonl");
    CHECK_FALSE(chibillm::parse_cli_options(4, args).has_value());
    args[2] = serve;
    CHECK_FALSE(chibillm::parse_cli_options(3, args).has_value());
}
