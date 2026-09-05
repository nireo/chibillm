#include "qwen/qwen_model_runner.h"
#include "metal/metal_model_state.h"
#include "qwen/qwen_chat.h"
#include "text.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "model_format/safetensors.h"
#include "qwen/qwen_layer.h"
#include "tensor/embedding.h"
#include "tensor/output.h"

namespace chibillm {
namespace {

class qwen_text_decoder final : public text_decoder {
public:
    explicit qwen_text_decoder(const qwen_tokenizer& tokenizer)
        : tokenizer_(tokenizer)
    {}

    result<std::string, model_runner_errc>
    push(token_id token, bool final) override
    {
        auto bytes = tokenizer_.decode(std::span(&token, 1));
        if (!bytes)
            return fail(model_runner_errc::tokenizer_failure);
        pending_ += *bytes;
        const auto prefix = complete_utf8_prefix(pending_);
        if (!prefix || (final && *prefix != pending_.size()))
            return fail(model_runner_errc::tokenizer_failure);
        auto delta = pending_.substr(0, *prefix);
        pending_.erase(0, *prefix);
        return delta;
    }

private:
    const qwen_tokenizer& tokenizer_;
    std::string pending_;
};

template <typename Error>
void
log_load_failure(const char* stage, Error error)
{
    std::fprintf(stderr, "[model-load] %s failed (error=%u)\n", stage,
                 static_cast<unsigned>(error));
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
    kv_cache_config cache_config {
        .layer_count = config->layer_count,
        .block_count = kv_block_count,
        .block_size = kv_block_size,
        .kv_head_count = config->kv_head_count,
        .head_dimension = config->head_dimension,
    };
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
                               cache_config,        std::move(*tokenizer), std::move(info) };
}

qwen_model_runner::qwen_model_runner(metal_context context,
                                     qwen3_config config,
                                     qwen_weights weights,
                                     kv_cache_config cache_config,
                                     qwen_tokenizer tokenizer,
                                     model_info info)
    : context_(std::move(context))
    , config_(std::move(config))
    , weights_(std::move(weights))
    , cache_config_(cache_config)
    , tokenizer_(std::move(tokenizer))
    , info_(std::move(info))
{}

result<std::unique_ptr<model_state>, model_runner_errc>
qwen_model_runner::make_state(scheduler_config config) const
{
    if (config.kv_block_size != cache_config_.block_size
        || config.kv_block_count > cache_config_.block_count) {
        return fail(model_runner_errc::inconsistent_batch);
    }
    auto geometry = cache_config_;
    geometry.block_count = config.kv_block_count;
    auto state = metal_model_state::make(context_, geometry);
    if (!state)
        return fail(model_runner_errc::backend_failure);
    return std::move(*state);
}

std::unique_ptr<text_decoder>
qwen_model_runner::make_decoder() const
{
    return std::make_unique<qwen_text_decoder>(tokenizer_);
}

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
    auto prompt = format_qwen_chat(messages);
    if (!prompt)
        return fail(prompt.error());
    auto tokens = tokenizer_.encode(*prompt);
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
qwen_model_runner::execute(const model_batch& batch, model_state& state)
{
    auto* paged_state = dynamic_cast<metal_model_state*>(&state);
    if (!paged_state)
        return fail(model_runner_errc::inconsistent_batch);
    auto& cache = paged_state->cache();
    auto metadata = prepare_paged_batch(batch, config_.max_position_embeddings, cache.block_count(),
                                        cache.block_size());
    if (!metadata) {
        return fail(metadata.error() == model_batch_errc::empty_batch
                        ? model_runner_errc::empty_batch
                        : model_runner_errc::inconsistent_batch);
    }

    // every kernel of the forward pass is encoded into one command buffer and
    // awaited exactly once at the end; per-op waits would dominate wall time.
    compute_pass pass(context_);
    auto pass_started = pass.begin();
    if (!pass_started) {
        return fail(model_runner_errc::backend_failure);
    }
    auto hidden_states = embed_tokens(context_, weights_.token_embedding, batch.tokens);
    if (!hidden_states) {
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
                                        cache);
    if (!final_hidden) {
        return fail(model_runner_errc::backend_failure);
    }
    if (metadata->logits_indices.empty()) {
        if (!pass.finish())
            return fail(model_runner_errc::backend_failure);
        return std::vector<token_id> {};
    }
    auto encoded_tokens =
        encode_greedy(context_, weights_.final_norm, weights_.output, config_.rms_epsilon,
                      *final_hidden, metadata->logits_indices);
    if (!encoded_tokens) {
        return fail(model_runner_errc::backend_failure);
    }
    auto pass_finished = pass.finish();
    if (!pass_finished) {
        return fail(model_runner_errc::backend_failure);
    }

    // The pass writes only one int32 per sequence for the CPU to read.
    return read_greedy(*encoded_tokens);
}

} // namespace chibillm
