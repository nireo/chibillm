#pragma once

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <string>

#include "result.h"

namespace chibillm {

struct cli_options {
    static constexpr std::size_t kv_block_size = 16;

    std::filesystem::path model_directory { "qwen_model" };
    std::size_t context_length { 32768 };
    std::size_t max_tokens { 8192 };
    bool serve { false };
    bool help { false };
};

[[nodiscard]] result<cli_options, std::string> parse_cli_options(int argc, char** argv);
void print_usage(std::ostream& output);

} // namespace chibillm
