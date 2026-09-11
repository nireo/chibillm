#include "application.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>

#include "cli_options.h"
#include "model_factory.h"
#include "repl.h"
#include "server.h"

namespace chibillm {
namespace {

std::string
load_text(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

int
run_server(model_runner& runner, std::size_t kv_block_count, std::size_t max_tokens)
{
    auto server = openai_server::make(runner,
                                      {
                                          .host = "127.0.0.1",
                                          .port = 8000,
                                          .max_sequences = 4,
                                          .max_pending_requests = 64,
                                          .max_batch_tokens = 128,
                                          .kv_block_count = kv_block_count,
                                          .kv_block_size = cli_options::kv_block_size,
                                          .default_max_completion_tokens = max_tokens,
                                      });
    if (!server) {
        std::cerr << "failed to start the HTTP server\n";
        return 1;
    }
    std::cerr << "listening on http://127.0.0.1:" << (*server)->port() << "/v1\n";
    if (!(*server)->run()) {
        std::cerr << "HTTP server failed\n";
        return 1;
    }
    return 0;
}

} // namespace

int
run_application(const cli_options& settings, const std::filesystem::path& shader_path)
{
    constexpr auto kv_block_size = cli_options::kv_block_size;
    const auto& model_directory = settings.model_directory;
    const auto requested_blocks = settings.context_length / kv_block_size;
    const auto shader_source = load_text(shader_path);
    if (shader_source.empty()) {
        std::cerr << "failed to load Metal shaders\n";
        return 1;
    }
    const auto load_started = std::chrono::steady_clock::now();
    auto model_id = model_directory.lexically_normal().filename().string();
    if (model_id.empty()) {
        model_id = "chibillm-qwen";
    }
    auto runner = load_model(model_directory, shader_source, requested_blocks, kv_block_size,
                             std::move(model_id));
    if (!runner) {
        std::cerr
            << (runner.error() == model_load_errc::unsupported_architecture
                    ? "unsupported model architecture in "
                    : "failed to load model from ")
            << model_directory
            << '\n';
        return 1;
    }
    const auto context_length = (*runner)->info().max_context_tokens;
    if (context_length < 2) {
        std::cerr << "model context is too small for generation\n";
        return 1;
    }
    const auto kv_block_count = (context_length + kv_block_size - 1) / kv_block_size;
    const auto max_tokens = std::min(settings.max_tokens, context_length - 1);
    const auto load_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - load_started).count();
    std::cerr
        << std::fixed
        << std::setprecision(2)
        << "[perf] model loaded in "
        << load_time
        << " s\n";
    std::cerr
        << "[limits] context "
        << context_length
        << " tok | max output "
        << max_tokens
        << " tok\n";
    if (settings.serve)
        return run_server(**runner, kv_block_count, max_tokens);

    return run_repl(**runner,
                    { .max_sequences = 1,
                      .max_batch_tokens = 128,
                      .kv_block_count = kv_block_count,
                      .kv_block_size = kv_block_size,
                      .eos_token = (*runner)->info().eos_token },
                    max_tokens, settings.stream, settings.progress);
}

} // namespace chibillm
