#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "metal/metal_context.h"
#include "metal/metal_kv_cache.h"
#include "model_runner.h"
#include "qwen/qwen_config.h"
#include "qwen/qwen_tokenizer.h"
#include "qwen/qwen_weights.h"
#include "result.h"

namespace chibillm {

enum class qwen_model_runner_errc : std::uint8_t {
    config_load_failed,
    tokenizer_load_failed,
    metal_context_creation_failed,
    weights_open_failed,
    weights_load_failed,
    cache_creation_failed,
};

// Connects scheduler batches to one resident Qwen model and its persistent KV cache.
class qwen_model_runner final : public model_runner {
public:
    [[nodiscard]] static result<qwen_model_runner, qwen_model_runner_errc>
    make(const std::filesystem::path& model_directory,
         std::string_view shader_source,
         std::size_t kv_block_count,
         std::size_t kv_block_size,
         std::string model_id = "chibillm-qwen");

    qwen_model_runner(const qwen_model_runner&) = delete;
    qwen_model_runner& operator=(const qwen_model_runner&) = delete;
    qwen_model_runner(qwen_model_runner&&) noexcept = default;
    qwen_model_runner& operator=(qwen_model_runner&&) noexcept = default;

    [[nodiscard]] const qwen_config& config() const noexcept;
    [[nodiscard]] const model_info& info() const noexcept override;

    [[nodiscard]] result<std::vector<token_id>, model_runner_errc>
    encode_chat(std::span<const chat_message> messages) override;

    [[nodiscard]] result<std::string, model_runner_errc>
    decode(std::span<const token_id> tokens) const override;

    [[nodiscard]] result<std::vector<token_id>, model_runner_errc>
    execute(const model_batch& batch) override;

private:
    qwen_model_runner(metal_context context,
                      qwen_config config,
                      qwen_weights weights,
                      metal_kv_cache cache,
                      qwen_tokenizer tokenizer,
                      model_info info);

    metal_context context_;
    qwen_config config_;
    qwen_weights weights_;
    metal_kv_cache cache_;
    qwen_tokenizer tokenizer_;
    model_info info_;
};

} // namespace chibillm
