#pragma once

#include "error.h"
#include "model_runner.h"
#include <filesystem>

namespace chibillm {
enum class model_load_errc {
    invalid_config,
    unsupported_architecture,
    load_failed
};

[[nodiscard]] inline std::string_view
error_name(model_load_errc code) noexcept
{
    static constexpr std::array names {
        "model_load.invalid_config",
        "model_load.unsupported_architecture",
        "model_load.load_failed",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "model_load.unknown_error";
}

using model_load_error = error<model_load_errc>;
result<std::unique_ptr<model_runner>, model_load_error>
load_model(const std::filesystem::path& directory,
           std::string_view kernel_source,
           std::size_t block_count,
           std::size_t block_size,
           std::string id);
} // namespace chibillm
