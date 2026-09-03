#include "qwen/qwen_model_runner.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "model_format/safetensors.h"
#include "qwen/qwen_embedding.h"
#include "qwen/qwen_layer.h"
#include "qwen/qwen_output.h"

namespace chibillm {
namespace {

template <typename Error>
void
log_load_failure(const char* stage, Error error)
{
    std::fprintf(stderr, "[model-load] %s failed (error=%u)\n", stage,
                 static_cast<unsigned>(error));
}

struct batch_metadata {
    std::vector<std::uint32_t> slots;
    std::vector<std::uint32_t> block_table;
    std::vector<std::uint32_t> table_offsets;
    std::vector<std::uint32_t> table_lengths;
    std::vector<std::size_t> logits_indices;
};

result<batch_metadata, model_runner_errc>
prepare_metadata(const model_batch& batch, const qwen3_config& config, const metal_kv_cache& cache)
{
    if (batch.empty()) {
        return fail(model_runner_errc::empty_batch);
    }
    if (batch.tokens.size() != batch.positions.size()
        || batch.kv_block_size != cache.block_size()) {
        return fail(model_runner_errc::inconsistent_batch);
    }

    batch_metadata metadata;
    metadata.slots.reserve(batch.tokens.size());
    metadata.table_offsets.reserve(batch.tokens.size());
    metadata.table_lengths.reserve(batch.tokens.size());
    metadata.logits_indices.reserve(batch.items.size());

    std::size_t expected_token_offset = 0;
    constexpr auto max_u32 = std::numeric_limits<std::uint32_t>::max();
    for (const auto& item : batch.items) {
        if (item.token_count == 0
            || item.token_offset != expected_token_offset
            || item.token_count > batch.tokens.size() - expected_token_offset
            || item.logits_index < item.token_offset
            || item.logits_index >= item.token_offset + item.token_count
            || item.block_table.empty()
            || metadata.block_table.size() > max_u32
            || item.block_table.size() > max_u32 - metadata.block_table.size()) {
            return fail(model_runner_errc::inconsistent_batch);
        }

        const auto table_offset = static_cast<std::uint32_t>(metadata.block_table.size());
        const auto table_length = static_cast<std::uint32_t>(item.block_table.size());
        for (const auto physical_block : item.block_table) {
            if (physical_block >= cache.block_count()) {
                return fail(model_runner_errc::inconsistent_batch);
            }
            metadata.block_table.push_back(physical_block);
        }

        for (std::size_t row = item.token_offset; row < item.token_offset + item.token_count;
             ++row) {
            const auto position = batch.positions[row];
            if (position >= config.max_position_embeddings) {
                return fail(model_runner_errc::inconsistent_batch);
            }
            const auto logical_block = static_cast<std::size_t>(position) / cache.block_size();
            if (logical_block >= item.block_table.size()) {
                return fail(model_runner_errc::inconsistent_batch);
            }

            const auto physical_block = item.block_table[logical_block];
            const auto token_offset = static_cast<std::size_t>(position) % cache.block_size();
            const auto slot =
                static_cast<std::size_t>(physical_block) * cache.block_size() + token_offset;
            if (slot > max_u32) {
                return fail(model_runner_errc::inconsistent_batch);
            }

            metadata.slots.push_back(static_cast<std::uint32_t>(slot));
            metadata.table_offsets.push_back(table_offset);
            metadata.table_lengths.push_back(table_length);
        }

        metadata.logits_indices.push_back(item.logits_index);
        expected_token_offset += item.token_count;
    }

    if (expected_token_offset != batch.tokens.size()) {
        return fail(model_runner_errc::inconsistent_batch);
    }
    return metadata;
}

} // namespace

result<qwen_model_runner, qwen_model_runner_errc>
qwen_model_runner::make(const std::filesystem::path& model_directory,
                        std::string_view shader_source,
                        std::size_t kv_block_count,
                        std::size_t kv_block_size,
                        std::string model_id)
{
    if (kv_block_count == 0
        || kv_block_size == 0
        || kv_block_count > std::numeric_limits<std::size_t>::max() / kv_block_size) {
        return fail(qwen_model_runner_errc::cache_creation_failed);
    }
    auto config = load_qwen3_config(model_directory / "config.json");
    if (!config) {
        log_load_failure("config", config.error());
        return fail(qwen_model_runner_errc::config_load_failed);
    }
    auto tokenizer = qwen_tokenizer::load(model_directory);
    if (!tokenizer) {
        log_load_failure("tokenizer", tokenizer.error());
        return fail(qwen_model_runner_errc::tokenizer_load_failed);
    }
    auto file = safetensors_file::open(model_directory / "model.safetensors");
    if (!file) {
        log_load_failure("safetensors", file.error());
        return fail(qwen_model_runner_errc::weights_open_failed);
    }
    auto context = metal_context::make(shader_source);
    if (!context) {
        std::fprintf(stderr, "[model-load] Metal context failed: %s\n",
                     context.error().message.c_str());
        return fail(qwen_model_runner_errc::metal_context_creation_failed);
    }
    auto cache = metal_kv_cache::make(*context,
                                      {
                                          .layer_count = config->layer_count,
                                          .block_count = kv_block_count,
                                          .block_size = kv_block_size,
                                          .kv_head_count = config->kv_head_count,
                                          .head_dimension = config->head_dimension,
                                      });
    if (!cache) {
        log_load_failure("KV cache", cache.error());
        return fail(qwen_model_runner_errc::cache_creation_failed);
    }
    auto weights = load_qwen_weights(*context, *file, *config);
    if (!weights) {
        log_load_failure("weights", weights.error());
        return fail(qwen_model_runner_errc::weights_load_failed);
    }

    const auto context_tokens =
        std::min(config->max_position_embeddings, kv_block_count * kv_block_size);
    model_info info {
        .id = std::move(model_id),
        .max_context_tokens = context_tokens,
        .eos_token = config->eos_token_id,
    };
    return qwen_model_runner { std::move(*context), std::move(*config),    std::move(*weights),
                               std::move(*cache),   std::move(*tokenizer), std::move(info) };
}

qwen_model_runner::qwen_model_runner(metal_context context,
                                     qwen3_config config,
                                     qwen_weights weights,
                                     metal_kv_cache cache,
                                     qwen_tokenizer tokenizer,
                                     model_info info)
    : context_(std::move(context))
    , config_(std::move(config))
    , weights_(std::move(weights))
    , cache_(std::move(cache))
    , tokenizer_(std::move(tokenizer))
    , info_(std::move(info))
{}

const qwen3_config&
qwen_model_runner::config() const noexcept
{
    return config_;
}

const model_info&
qwen_model_runner::info() const noexcept
{
    return info_;
}

result<std::vector<token_id>, model_runner_errc>
qwen_model_runner::encode_chat(std::span<const chat_message> messages)
{
    if (messages.empty()) {
        return fail(model_runner_errc::invalid_chat);
    }

    std::string prompt;
    for (const auto& message : messages) {
        if (message.role != "developer"
            && message.role != "system"
            && message.role != "user"
            && message.role != "assistant") {
            return fail(model_runner_errc::invalid_chat);
        }
        const auto role = message.role == "developer" ? "system" : message.role;
        prompt += "<|im_start|>" + role + "\n" + message.content + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";

    auto tokens = tokenizer_.encode(prompt);
    if (!tokens) {
        return fail(model_runner_errc::tokenizer_failure);
    }
    return std::move(*tokens);
}

result<std::string, model_runner_errc>
qwen_model_runner::decode(std::span<const token_id> tokens) const
{
    auto text = tokenizer_.decode(tokens);
    if (!text) {
        return fail(model_runner_errc::tokenizer_failure);
    }
    return std::move(*text);
}

result<std::vector<token_id>, model_runner_errc>
qwen_model_runner::execute(const model_batch& batch)
{
    auto metadata = prepare_metadata(batch, config_, cache_);
    if (!metadata) {
        return fail(metadata.error());
    }

    // every kernel of the forward pass is encoded into one command buffer and
    // awaited exactly once at the end; per-op waits would dominate wall time.
    auto pass_started = context_.begin_compute_pass();
    if (!pass_started) {
        return fail(model_runner_errc::backend_failure);
    }
    auto hidden_states = embed_qwen_tokens(context_, weights_, batch.tokens);
    if (!hidden_states) {
        context_.abort_compute_pass();
        return fail(model_runner_errc::backend_failure);
    }
    auto final_hidden = run_qwen_layers(context_, config_, weights_, std::move(*hidden_states),
                                        {
                                            .positions = batch.positions,
                                            .slots = metadata->slots,
                                            .block_table = metadata->block_table,
                                            .block_table_offsets = metadata->table_offsets,
                                            .block_table_lengths = metadata->table_lengths,
                                        },
                                        cache_);
    if (!final_hidden) {
        context_.abort_compute_pass();
        return fail(model_runner_errc::backend_failure);
    }
    auto encoded_tokens =
        encode_qwen_greedy(context_, config_, weights_, *final_hidden, metadata->logits_indices);
    if (!encoded_tokens) {
        context_.abort_compute_pass();
        return fail(model_runner_errc::backend_failure);
    }
    auto pass_finished = context_.end_compute_pass();
    if (!pass_finished) {
        return fail(model_runner_errc::backend_failure);
    }

    // The pass writes only one int32 per sequence for the CPU to read.
    return read_qwen_greedy(*encoded_tokens);
}

} // namespace chibillm
