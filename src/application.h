#pragma once

#include <filesystem>

namespace chibillm {

struct cli_options;

int run_application(const cli_options& settings, const std::filesystem::path& shader_path);

} // namespace chibillm
