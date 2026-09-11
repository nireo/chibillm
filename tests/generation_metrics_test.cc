#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <iomanip>
#include <sstream>

#include "generation_metrics.h"

TEST_CASE("metrics separate prefill, decode execution and inter-token wall time")
{
    chibillm::generation_metrics m;
    m.admission_at = 0.125;
    m.record_batch(true, 128, 0.25);
    m.record_batch(true, 32, 0.125);
    m.record_token(0.5);
    m.record_batch(false, 0, 0.0625);
    m.record_token(0.625); // Includes output/loop work, unlike the decode step.
    m.record_batch(false, 0, 0.125);
    m.record_token(0.875);
    m.summarize();
    CHECK(m.prefill_batches == 2);
    CHECK(m.processed_prompt_tokens == 160);
    CHECK(m.output_tokens == 3);
    CHECK(m.prefill_seconds == 0.375);
    CHECK(m.decode_seconds == 0.1875);
    CHECK(m.engine_ttft_seconds() == 0.375);
    CHECK(*m.prefill_tokens_per_second() == doctest::Approx(160 / 0.375));
    CHECK(*m.decode_tokens_per_second() == doctest::Approx(2 / 0.375));
    CHECK(m.decode_step.samples == 2);
    CHECK(m.decode_step.p50 == 0.0625);
    CHECK(m.decode_step.p95 == 0.125);
    CHECK(m.inter_token.samples == 2);
    CHECK(m.inter_token.p50 == 0.125);
    CHECK(m.inter_token.p95 == 0.25);
}

TEST_CASE("missing measurements serialize as null rather than zero or infinity")
{
    chibillm::generation_metrics m;
    m.summarize();
    CHECK_FALSE(m.engine_ttft_seconds());
    CHECK_FALSE(m.prefill_tokens_per_second());
    CHECK_FALSE(m.decode_tokens_per_second());
    std::ostringstream output;
    output << std::fixed << std::setprecision(2); // JSON ignores presentation formatting.
    m.tokenization_seconds = 0.123456789012345;
    REQUIRE(chibillm::write_metrics_jsonl(output, m, { "test", 16, 99 }, {}, "session", 1, true,
                                          false));
    const auto row = nlohmann::json::parse(output.str());
    CHECK(row["schema_version"] == 1);
    CHECK(row["engine_ttft_seconds"].is_null());
    CHECK(row["first_text_seconds"].is_null());
    CHECK(row["prefill_tokens_per_second"].is_null());
    CHECK(row["decode_tokens_per_second"].is_null());
    CHECK(row["decode_step"]["samples"] == 0);
    CHECK(row["decode_step"]["p95_seconds"].is_null());
    CHECK(row["tokenization_seconds"].get<double>() == m.tokenization_seconds);
    CHECK_FALSE(row.contains("prompt"));
    CHECK_FALSE(row.contains("response"));
}

TEST_CASE("one output token has no decode or inter-token samples")
{
    chibillm::generation_metrics m;
    m.record_batch(true, 1, 0.125);
    m.record_token(0.25);
    m.summarize();
    CHECK(m.engine_ttft_seconds() == 0.25);
    CHECK_FALSE(m.first_text_seconds);
    CHECK_FALSE(m.decode_tokens_per_second());
    CHECK(m.decode_step.samples == 0);
    CHECK(m.inter_token.samples == 0);
}
