#pragma once

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string_view>
#include <vector>

#include "model_runner.h"
#include "scheduler.h"

namespace chibillm {

struct latency_summary {
    std::size_t samples = 0;
    std::optional<double> p50;
    std::optional<double> p95;
};

// All durations are seconds. Event timestamps are relative to request submission.
// Accounting accepts timestamps explicitly so boundaries can be tested without sleeps.
struct generation_metrics {
    std::size_t prompt_tokens = 0;
    std::size_t processed_prompt_tokens = 0;
    std::size_t output_tokens = 0; // Includes EOS, even if it produces no text.
    std::size_t output_bytes = 0;  // Successfully decoded answer bytes, not terminal framing.
    std::size_t requested_tokens = 0;
    std::size_t token_budget = 0;
    std::size_t prefill_batches = 0;
    double tokenization_seconds = 0;
    double prefill_seconds = 0;
    double decode_seconds = 0;
    double text_decode_seconds = 0;
    double output_write_seconds = 0;
    double end_to_end_seconds = 0;
    double admission_at = 0;
    std::optional<double> first_token_at;
    std::optional<double> first_text_seconds;
    std::optional<double> last_token_at;
    std::string_view status = "error";
    std::string_view stage = "tokenization";
    std::string_view stop = "none";
    latency_summary decode_step;
    latency_summary inter_token;

    void record_batch(bool prefill, std::size_t prompt_count, double seconds);
    void record_token(double at);
    void summarize();
    [[nodiscard]] std::optional<double> engine_ttft_seconds() const;
    [[nodiscard]] std::optional<double> prefill_tokens_per_second() const;
    [[nodiscard]] std::optional<double> decode_tokens_per_second() const;

private:
    std::vector<double> decode_samples_;
    std::vector<double> inter_token_samples_;
};

// Serializes only at request completion; never includes prompt or response text.
bool write_metrics_jsonl(std::ostream& output,
                         const generation_metrics& metrics,
                         const model_info& model,
                         scheduler_config config,
                         std::string_view session_id,
                         std::size_t request_id,
                         bool stream,
                         bool progress);

} // namespace chibillm
