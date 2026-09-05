#pragma once
#include "model_runner.h"
#include <filesystem>

namespace chibillm {
enum class model_load_errc {
    invalid_config,
    unsupported_architecture,
    load_failed
};
result<std::unique_ptr<model_runner>, model_load_errc>
load_model(const std::filesystem::path& directory,
           std::string_view shaders,
           std::size_t block_count,
           std::size_t block_size,
           std::string id);
} // namespace chibillm
