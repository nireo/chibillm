#include "qwen/qwen_model_runner.h"
#include "metal/metal_model_state.h"
#include "qwen/qwen3_5_model_state.h"
#include "qwen/qwen_chat.h"
#include "text.h"

#include <algorithm>
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

    result<std::string, model_runner_error>
    push(token_id token, bool final) override
    {
        auto bytes = tokenizer_.decode(std::span(&token, 1));
        if (!bytes)
            return fail(model_runner_errc::tokenizer_failure, bytes.error(), "token decode");
        pending_ += *bytes;
        const auto prefix = complete_utf8_prefix(pending_);
        if (!prefix || (final && *prefix != pending_.size()))
            return fail(model_runner_errc::tokenizer_failure, "invalid or incomplete UTF-8",
                        "text decode");
        auto delta = pending_.substr(0, *prefix);
        pending_.erase(0, *prefix);
        return delta;
    }

private:
    const qwen_tokenizer& tokenizer_;
    std::string pending_;
};

// Finish the forward pass once, then read only one token per requested row.
result<std::vector<token_id>, model_runner_error>
finish_greedy(compute_pass& pass,
              const metal_context& context,
              const metal_tensor& norm,
              const metal_tensor& vocabulary,
              float epsilon,
              const metal_tensor& hidden_states,
              std::span<const std::size_t> logits_indices,
              bool zero_centered = false)
{
    if (logits_indices.empty()) {
        auto finished = pass.finish();
        if (!finished)
            return fail(model_runner_errc::backend_failure, finished.error(),
                        "finish compute pass");
        return std::vector<token_id> {};
    }
    auto encoded = encode_greedy(context, norm, vocabulary, epsilon, hidden_states, logits_indices,
                                 zero_centered);
    if (!encoded)
        return fail(model_runner_errc::backend_failure, encoded.error(), "greedy output");
    auto finished = pass.finish();
    if (!finished)
        return fail(model_runner_errc::backend_failure, finished.error(), "finish compute pass");
    return read_greedy(*encoded);
}

} // namespace

result<qwen_model_runner, qwen_model_runner_error>
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
        return fail(qwen_model_runner_errc::config_load_failed, config.error(), "config");
    }
    auto tokenizer = qwen_tokenizer::load(model_directory);
    if (!tokenizer) {
        return fail(qwen_model_runner_errc::tokenizer_load_failed, tokenizer.error(), "tokenizer");
    }
    auto file = safetensors_file::open(model_directory / "model.safetensors");
    if (!file) {
        return fail(qwen_model_runner_errc::weights_open_failed, file.error(), "safetensors");
    }
    auto context = metal_context::make(shader_source);
    if (!context) {
        return fail(qwen_model_runner_errc::metal_context_creation_failed, context.error());
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
        return fail(qwen_model_runner_errc::weights_load_failed, weights.error(), "weights");
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

result<std::unique_ptr<model_state>, model_runner_error>
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
        return fail(model_runner_errc::backend_failure, state.error(), "create model state");
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

result<std::vector<token_id>, model_runner_error>
qwen_model_runner::encode_chat(std::span<const chat_message> messages)
{
    auto prompt = format_qwen_chat(messages);
    if (!prompt)
        return fail(prompt.error());
    auto tokens = tokenizer_.encode(*prompt);
    if (!tokens) {
        return fail(model_runner_errc::tokenizer_failure, tokens.error(), "chat encode");
    }
    return std::move(*tokens);
}

result<std::string, model_runner_error>
qwen_model_runner::decode(std::span<const token_id> tokens) const
{
    auto text = tokenizer_.decode(tokens);
    if (!text) {
        return fail(model_runner_errc::tokenizer_failure, text.error(), "text decode");
    }
    return std::move(*text);
}

result<std::vector<token_id>, model_runner_error>
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
                        : model_runner_errc::inconsistent_batch,
                    metadata.error(), "prepare batch");
    }

    // every kernel of the forward pass is encoded into one command buffer and
    // awaited exactly once at the end; per-op waits would dominate wall time.
    compute_pass pass(context_);
    auto pass_started = pass.begin();
    if (!pass_started) {
        return fail(model_runner_errc::backend_failure, pass_started.error(), "begin compute pass");
    }
    auto hidden_states = embed_tokens(context_, weights_.token_embedding, batch.tokens);
    if (!hidden_states) {
        return fail(model_runner_errc::backend_failure, hidden_states.error(), "embedding");
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
        return fail(model_runner_errc::backend_failure, final_hidden.error(), "layers");
    }
    return finish_greedy(pass, context_, weights_.final_norm, weights_.output, config_.rms_epsilon,
                         *final_hidden, metadata->logits_indices);
}

result<qwen3_5_model_runner, qwen_model_runner_error>
qwen3_5_model_runner::make(const std::filesystem::path& model_directory,
                           std::string_view shader_source,
                           std::size_t kv_block_count,
                           std::size_t kv_block_size,
                           std::string model_id)
{
    if (!kv_block_count
        || !kv_block_size
        || kv_block_count > std::numeric_limits<std::size_t>::max() / kv_block_size)
        return fail(qwen_model_runner_errc::cache_creation_failed);
    auto config = load_qwen3_5_config(model_directory / "config.json");
    if (!config) {
        return fail(qwen_model_runner_errc::config_load_failed, config.error(), "config");
    }
    auto tokenizer = qwen_tokenizer::load(model_directory);
    if (!tokenizer) {
        return fail(qwen_model_runner_errc::tokenizer_load_failed, tokenizer.error(), "tokenizer");
    }
    auto file = safetensors_file::open_model(model_directory);
    if (!file) {
        return fail(qwen_model_runner_errc::weights_open_failed, file.error(), "safetensors");
    }
    auto context = metal_context::make(shader_source);
    if (!context) {
        return fail(qwen_model_runner_errc::metal_context_creation_failed, context.error());
    }
    auto weights = load_qwen3_5_weights(*context, *file, *config);
    if (!weights) {
        return fail(qwen_model_runner_errc::weights_load_failed, weights.error(), "weights");
    }
    model_info info {
        .id = std::move(model_id),
        .max_context_tokens =
            std::min(config->max_position_embeddings, kv_block_count * kv_block_size),
        .eos_token = config->eos_token_id,
    };
    return qwen3_5_model_runner { std::move(*context), std::move(*config), std::move(*weights),
                                  kv_block_count,      kv_block_size,      std::move(*tokenizer),
                                  std::move(info) };
}

qwen3_5_model_runner::qwen3_5_model_runner(metal_context context,
                                           qwen3_5_config config,
                                           qwen3_5_weights weights,
                                           std::size_t block_count,
                                           std::size_t block_size,
                                           qwen_tokenizer tokenizer,
                                           model_info info)
    : context_(std::move(context))
    , config_(std::move(config))
    , weights_(std::move(weights))
    , block_count_(block_count)
    , block_size_(block_size)
    , tokenizer_(std::move(tokenizer))
    , info_(std::move(info))
{}

result<std::unique_ptr<model_state>, model_runner_error>
qwen3_5_model_runner::make_state(scheduler_config config) const
{
    if (config.kv_block_size != block_size_ || config.kv_block_count > block_count_)
        return fail(model_runner_errc::inconsistent_batch);
    auto state = qwen3_5_model_state::make(context_, config_, config.kv_block_count, block_size_);
    if (!state)
        return fail(model_runner_errc::backend_failure, state.error(), "create model state");
    return std::move(*state);
}

std::unique_ptr<text_decoder>
qwen3_5_model_runner::make_decoder() const
{
    return std::make_unique<qwen_text_decoder>(tokenizer_);
}

const model_info&
qwen3_5_model_runner::info() const noexcept
{
    return info_;
}

result<std::vector<token_id>, model_runner_error>
qwen3_5_model_runner::encode_chat(std::span<const chat_message> messages)
{
    auto prompt = format_qwen_chat(messages);
    if (!prompt)
        return fail(prompt.error());
    auto tokens = tokenizer_.encode(*prompt);
    if (!tokens)
        return fail(model_runner_errc::tokenizer_failure, tokens.error(), "chat encode");
    return std::move(*tokens);
}

result<std::string, model_runner_error>
qwen3_5_model_runner::decode(std::span<const token_id> tokens) const
{
    auto text = tokenizer_.decode(tokens);
    if (!text)
        return fail(model_runner_errc::tokenizer_failure, text.error(), "text decode");
    return std::move(*text);
}

result<std::vector<token_id>, model_runner_error>
qwen3_5_model_runner::execute(const model_batch& batch, model_state& state)
{
    auto* hybrid = dynamic_cast<qwen3_5_model_state*>(&state);
    if (!hybrid)
        return fail(model_runner_errc::inconsistent_batch);
    auto metadata = prepare_paged_batch(batch, config_.max_position_embeddings,
                                        hybrid->cache().block_count(), hybrid->block_size());
    if (!metadata)
        return fail(metadata.error() == model_batch_errc::empty_batch
                        ? model_runner_errc::empty_batch
                        : model_runner_errc::inconsistent_batch,
                    metadata.error(), "prepare batch");

    // The engine has already begun the state transaction. On any early return,
    // the pass destructor drains encoded writes before the engine aborts it.
    compute_pass pass(context_);
    auto started = pass.begin();
    if (!started)
        return fail(model_runner_errc::backend_failure, started.error(), "begin compute pass");
    auto hidden = embed_tokens(context_, weights_.token_embedding, batch.tokens);
    if (!hidden)
        return fail(model_runner_errc::backend_failure, hidden.error(), "embedding");
    auto output =
        run_qwen3_5_layers(context_, config_, weights_, std::move(*hidden), batch, *hybrid);
    if (!output)
        return fail(model_runner_errc::backend_failure, output.error(), "layers");
    return finish_greedy(pass, context_, weights_.final_norm, weights_.token_embedding,
                         config_.rms_epsilon, *output, metadata->logits_indices, true);
}

} // namespace chibillm
