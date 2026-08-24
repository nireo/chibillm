#include "inference_engine.h"

#include <utility>

#include "model_batch.h"

namespace chibillm {

result<inference_engine, inference_engine_errc>
inference_engine::make(scheduler_config config, model_runner &runner) {
  auto scheduler_result = scheduler::make(config);
  if (!scheduler_result) {
    return fail(inference_engine_errc::scheduler_creation_failed);
  }

  return inference_engine{std::move(*scheduler_result), runner};
}

inference_engine::inference_engine(scheduler scheduler,
                                   model_runner &runner) noexcept
    : scheduler_(std::move(scheduler)), runner_(&runner) {}

bool inference_engine::is_finished() const noexcept {
  return scheduler_.is_finished();
}

bool inference_engine::has_in_flight_batch() const noexcept {
  return scheduler_.has_in_flight_batch();
}

const seq *inference_engine::find_sequence(seq_id id) const noexcept {
  return scheduler_.find_sequence(id);
}

result<void, inference_engine_errc> inference_engine::add(seq sequence) {
  auto added = scheduler_.add(std::move(sequence));
  if (!added) {
    return fail(inference_engine_errc::sequence_add_failed);
  }

  return {};
}

result<void, inference_engine_errc> inference_engine::step() {
  auto scheduled = scheduler_.schedule();
  if (!scheduled) {
    return fail(inference_engine_errc::scheduling_failed);
  }

  auto batch = build_model_batch(*scheduled, scheduler_);
  if (!batch) {
    return fail_after_abort(*scheduled,
                            inference_engine_errc::model_batch_build_failed);
  }

  auto sampled_tokens = runner_->execute(*batch);
  if (!sampled_tokens) {
    return fail_after_abort(*scheduled,
                            inference_engine_errc::model_execution_failed);
  }

  if (sampled_tokens->size() != scheduled->items.size()) {
    return fail_after_abort(
        *scheduled, inference_engine_errc::runner_result_count_mismatch);
  }

  auto completed = scheduler_.complete(*scheduled, *sampled_tokens);
  if (!completed) {
    return fail(inference_engine_errc::batch_completion_failed);
  }

  return {};
}

result<void, inference_engine_errc>
inference_engine::fail_after_abort(const scheduled_batch &batch,
                                   inference_engine_errc error) {
  auto aborted = scheduler_.abort(batch);
  if (!aborted) {
    return fail(inference_engine_errc::batch_abort_failed);
  }

  return fail(error);
}

} // namespace chibillm
