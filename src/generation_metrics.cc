#include "generation_metrics.h"

#include <algorithm>
#include <cmath>
#include <ostream>

#include <nlohmann/json.hpp>

namespace chibillm {
namespace {

latency_summary
summarize_samples(std::vector<double>& samples)
{
    if (samples.empty())
        return {};
    std::ranges::sort(samples);
    const auto percentile = [&](double fraction) {
        return samples[static_cast<std::size_t>(std::ceil(fraction * samples.size())) - 1];
    };
    return { samples.size(), percentile(0.50), percentile(0.95) };
}

nlohmann::json
nullable(std::optional<double> value)
{
    return value ? nlohmann::json(*value) : nlohmann::json(nullptr);
}

nlohmann::json
latency_json(const latency_summary& summary)
{
    return { { "samples", summary.samples },
             { "p50_seconds", nullable(summary.p50) },
             { "p95_seconds", nullable(summary.p95) } };
}

} // namespace

void
generation_metrics::record_batch(bool prefill, std::size_t prompt_count, double seconds)
{
    if (prefill) {
        ++prefill_batches;
        processed_prompt_tokens += prompt_count;
        prefill_seconds += seconds;
    } else {
        decode_seconds += seconds;
        decode_samples_.push_back(seconds);
    }
}

void
generation_metrics::record_token(double at)
{
    if (last_token_at)
        inter_token_samples_.push_back(at - *last_token_at);
    else
        first_token_at = at;
    last_token_at = at;
    ++output_tokens;
}

void
generation_metrics::summarize()
{
    decode_step = summarize_samples(decode_samples_);
    inter_token = summarize_samples(inter_token_samples_);
}

std::optional<double>
generation_metrics::engine_ttft_seconds() const
{
    return first_token_at ? std::optional(*first_token_at - admission_at) : std::nullopt;
}

std::optional<double>
generation_metrics::prefill_tokens_per_second() const
{
    return prefill_seconds > 0 ? std::optional(processed_prompt_tokens / prefill_seconds)
                               : std::nullopt;
}

std::optional<double>
generation_metrics::decode_tokens_per_second() const
{
    if (!first_token_at || !last_token_at || *last_token_at <= *first_token_at)
        return std::nullopt;
    return (output_tokens - 1) / (*last_token_at - *first_token_at);
}

bool
write_metrics_jsonl(std::ostream& output,
                    const generation_metrics& m,
                    const model_info& model,
                    scheduler_config config,
                    std::string_view session_id,
                    std::size_t request_id,
                    bool stream,
                    bool progress)
{
    const nlohmann::json record {
        { "schema_version", 1 },
        { "type", "request" },
        { "session_id", session_id },
        { "request_id", request_id },
        { "model", model.id },
        { "compiler", __VERSION__ },
#ifdef __OPTIMIZE__
        { "optimized_build", true },
#else
        { "optimized_build", false },
#endif
        { "stream", stream },
        { "progress", progress },
        { "context_tokens", model.max_context_tokens },
        { "max_batch_tokens", config.max_batch_tokens },
        { "kv_block_count", config.kv_block_count },
        { "kv_block_size", config.kv_block_size },
        { "status", m.status },
        { "stage", m.stage },
        { "stop_reason", m.stop },
        { "prompt_tokens", m.prompt_tokens },
        { "processed_prompt_tokens", m.processed_prompt_tokens },
        { "output_tokens", m.output_tokens },
        { "output_bytes", m.output_bytes },
        { "requested_tokens", m.requested_tokens },
        { "token_budget", m.token_budget },
        { "prefill_batches", m.prefill_batches },
        { "tokenization_seconds", m.tokenization_seconds },
        { "prefill_seconds", m.prefill_seconds },
        { "decode_engine_seconds", m.decode_seconds },
        { "engine_ttft_seconds", nullable(m.engine_ttft_seconds()) },
        { "first_text_seconds", nullable(m.first_text_seconds) },
        { "text_decode_seconds", m.text_decode_seconds },
        { "output_write_seconds", m.output_write_seconds },
        { "end_to_end_seconds", m.end_to_end_seconds },
        { "prefill_tokens_per_second", nullable(m.prefill_tokens_per_second()) },
        { "decode_tokens_per_second", nullable(m.decode_tokens_per_second()) },
        { "decode_step", latency_json(m.decode_step) },
        { "inter_token", latency_json(m.inter_token) },
    };
    const auto line = record.dump();
    output.write(line.data(), static_cast<std::streamsize>(line.size()));
    output.put('\n');
    output.flush();
    return static_cast<bool>(output);
}

} // namespace chibillm
