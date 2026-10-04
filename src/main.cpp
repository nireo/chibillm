#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include "cli_options.h"
#include "metal_kernel_source.h"
#include "model_factory.h"
#include "repl.h"
#include "server.h"

namespace chibillm {
namespace {

int
run_server(model_runner& runner,
           scheduler_config config,
           std::size_t max_tokens,
           std::size_t context_length)
{
    auto server = openai_server::make(
        runner,
        {
            .runtime = { .scheduler = config, .max_context_tokens = context_length },
            .default_max_completion_tokens = max_tokens,
        });
    if (!server) {
        std::cerr << "failed to start the HTTP server: " << describe_error(server.error()) << '\n';
        return 1;
    }
    std::cerr << "listening on http://127.0.0.1:" << (*server)->port() << "/v1\n";
    auto served = (*server)->run();
    if (!served) {
        std::cerr << "HTTP server failed: " << describe_error(served.error()) << '\n';
        return 1;
    }
    return 0;
}

int
run_application(const cli_options& settings)
{
    std::ofstream metrics_file;
    if (!settings.metrics_jsonl.empty()) {
        metrics_file.open(settings.metrics_jsonl, std::ios::app);
        if (!metrics_file) {
            std::cerr << "failed to open metrics file: " << settings.metrics_jsonl << '\n';
            return 1;
        }
    }
    constexpr auto kv_block_size = cli_options::kv_block_size;
    const auto& model_directory = settings.model_directory;
    const auto requested_blocks =
        settings.kv_cache_tokens.value_or(settings.context_length) / kv_block_size;
    // Loading records an upper bound; make_state allocates the actual shared pool below.
    const auto model_capacity_blocks =
        std::max(requested_blocks, settings.context_length / kv_block_size);
    const auto load_started = std::chrono::steady_clock::now();
    auto model_id = model_directory.lexically_normal().filename().string();
    if (model_id.empty()) {
        model_id = "chibillm-qwen";
    }
    auto runner = load_model(model_directory, metal_kernel_source, model_capacity_blocks,
                             kv_block_size, std::move(model_id), settings.quantization);
    if (!runner) {
        std::cerr
            << (runner.error() == model_load_errc::unsupported_architecture
                    ? "unsupported model architecture in "
                    : "failed to load model from ")
            << model_directory
            << ": "
            << describe_error(runner.error())
            << '\n';
        return 1;
    }
    const auto context_length =
        std::min(settings.context_length, (*runner)->info().max_context_tokens);
    if (context_length < 2) {
        std::cerr << "model context is too small for generation\n";
        return 1;
    }
    const auto kv_block_count = settings.kv_cache_tokens
        ? requested_blocks
        : (context_length + kv_block_size - 1) / kv_block_size;
    // A single chat can use larger matmul batches; serving keeps shorter
    // reservations so concurrent requests can take turns.
    const scheduler_config config {
        .max_sequences = settings.serve ? 4u : 1u,
        .max_batch_tokens = settings.serve ? 128u : 512u,
        .kv_block_count = kv_block_count,
        .kv_block_size = kv_block_size,
    };
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
        return run_server(**runner, config, max_tokens, context_length);

    return run_repl(**runner, config, settings.max_tokens, settings.stream, settings.progress,
                    metrics_file.is_open() ? &metrics_file : nullptr);
}

} // namespace
} // namespace chibillm

int
main(int argc, char** argv)
{
    auto settings = chibillm::parse_cli_options(argc, argv);
    if (!settings) {
        std::cerr << settings.error() << '\n';
        chibillm::print_usage(std::cerr);
        return 1;
    }
    if (settings->help) {
        chibillm::print_usage(std::cout);
        return 0;
    }
    return chibillm::run_application(*settings);
}
