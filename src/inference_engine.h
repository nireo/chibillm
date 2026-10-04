#pragma once

#include "error.h"

#include <cstdint>
#include <vector>

#include "model_runner.h"
#include "result.h"
#include "scheduler.h"

namespace chibillm {

enum class inference_engine_errc : std::uint8_t {
    scheduler_creation_failed,
    sequence_add_failed,
    scheduling_failed,
    model_batch_build_failed,
    model_execution_failed,
    runner_result_count_mismatch,
    batch_abort_failed,
    batch_completion_failed,
    sequence_cancel_failed,
    sequence_remove_failed,
};

[[nodiscard]] inline std::string_view
error_name(inference_engine_errc code) noexcept
{
    static constexpr std::array names {
        "inference_engine.scheduler_creation_failed",
        "inference_engine.sequence_add_failed",
        "inference_engine.scheduling_failed",
        "inference_engine.model_batch_build_failed",
        "inference_engine.model_execution_failed",
        "inference_engine.runner_result_count_mismatch",
        "inference_engine.batch_abort_failed",
        "inference_engine.batch_completion_failed",
        "inference_engine.sequence_cancel_failed",
        "inference_engine.sequence_remove_failed",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "inference_engine.unknown_error";
}

using inference_engine_error = error<inference_engine_errc>;

// coordinates scheduling, model execution, and sequence completion.
class inference_engine {
public:
    [[nodiscard]] static result<inference_engine, inference_engine_error>
    make(scheduler_config config, model_runner& runner);

    inference_engine(const inference_engine&) = delete;
    inference_engine& operator=(const inference_engine&) = delete;
    inference_engine(inference_engine&&) noexcept = default;
    inference_engine& operator=(inference_engine&&) noexcept = default;

    [[nodiscard]] bool is_finished() const noexcept;
    [[nodiscard]] bool has_in_flight_batch() const noexcept;
    [[nodiscard]] const seq* find_sequence(seq_id id) const noexcept;

    [[nodiscard]] result<void, inference_engine_error> add(seq sequence);
    [[nodiscard]] bool fits_cache(const seq& sequence) const noexcept;
    [[nodiscard]] result<admission_result, inference_engine_error> try_add(seq& sequence);
    [[nodiscard]] result<std::vector<sequence_update>, inference_engine_error> step();
    [[nodiscard]] result<void, inference_engine_error> cancel(seq_id id);
    [[nodiscard]] result<void, inference_engine_error> remove(seq_id id);

private:
    inference_engine(scheduler scheduler, model_runner& runner) noexcept;

    [[nodiscard]] std::unexpected<inference_engine_error>
    fail_after_abort(const scheduled_batch& batch, inference_engine_error error);

    scheduler scheduler_;
    model_runner* runner_;
};

} // namespace chibillm
