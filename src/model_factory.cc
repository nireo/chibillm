#include "model_factory.h"
#include "qwen/qwen_model_runner.h"
#include <fstream>
#include <nlohmann/json.hpp>

namespace chibillm {
result<std::unique_ptr<model_runner>, model_load_errc>
load_model(const std::filesystem::path& directory,
           std::string_view shaders,
           std::size_t block_count,
           std::size_t block_size,
           std::string id)
{
    std::ifstream input(directory / "config.json");
    if (!input)
        return fail(model_load_errc::invalid_config);
    auto config = nlohmann::json::parse(input, nullptr, false);
    if (!config.is_object()
        || !config.contains("model_type")
        || !config["model_type"].is_string()) {
        return fail(model_load_errc::invalid_config);
    }
    if (config["model_type"] == "qwen3") {
        auto runner =
            qwen_model_runner::make(directory, shaders, block_count, block_size, std::move(id));
        if (!runner)
            return fail(model_load_errc::load_failed);
        return std::make_unique<qwen_model_runner>(std::move(*runner));
    }
    return fail(model_load_errc::unsupported_architecture);
}
} // namespace chibillm
