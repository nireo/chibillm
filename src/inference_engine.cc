#include "inference_engine.h"

#include <utility>

#include "model_batch.h"

namespace chibillm {

result<inference_engine, inference_engine_error>
inference_engine::make(scheduler_config config, model_runner& runner)
{
    auto state = runner.make_state(config);
    if (!state)
        return fail(inference_engine_errc::scheduler_creation_failed, state.error(),
                    "create model state");
    auto scheduler_result = scheduler::make(config, runner.info().eos_token, std::move(*state));
    if (!scheduler_result) {
        return fail(inference_engine_errc::scheduler_creation_failed, scheduler_result.error());
    }

    return inference_engine { std::move(*scheduler_result), runner };
}

inference_engine::inference_engine(scheduler scheduler, model_runner& runner) noexcept
    : scheduler_(std::move(scheduler))
    , runner_(&runner)
{}

bool
inference_engine::is_finished() const noexcept
{
    return scheduler_.is_finished();
}

bool
inference_engine::has_in_flight_batch() const noexcept
{
    return scheduler_.has_in_flight_batch();
}

const seq*
inference_engine::find_sequence(seq_id id) const noexcept
{
    return scheduler_.find_sequence(id);
}

result<void, inference_engine_error>
inference_engine::add(seq sequence)
{
    auto added = scheduler_.add(std::move(sequence));
    if (!added) {
        return fail(inference_engine_errc::sequence_add_failed, added.error());
    }

    return {};
}

bool
inference_engine::fits_cache(const seq& sequence) const noexcept
{
    return scheduler_.fits_cache(sequence);
}

result<admission_result, inference_engine_error>
inference_engine::try_add(seq& sequence)
{
    auto admitted = scheduler_.try_add(sequence);
    if (!admitted)
        return fail(inference_engine_errc::sequence_add_failed, admitted.error());
    return *admitted;
}

result<std::vector<sequence_update>, inference_engine_error>
inference_engine::step()
{
    auto scheduled = scheduler_.schedule();
    if (!scheduled) {
        return fail(inference_engine_errc::scheduling_failed, scheduled.error());
    }

    auto batch = build_model_batch(*scheduled, scheduler_);
    if (!batch) {
        return fail_after_abort(
            *scheduled,
            { inference_engine_errc::model_batch_build_failed, describe_error(batch.error()) });
    }

    auto begun = scheduler_.begin_execution(*batch);
    if (!begun) {
        return fail_after_abort(*scheduled,
                                { inference_engine_errc::model_execution_failed,
                                  "begin batch: " + describe_error(begun.error()) });
    }
    auto sampled_tokens = runner_->execute(*batch, scheduler_.state());
    if (!sampled_tokens) {
        return fail_after_abort(*scheduled,
                                { inference_engine_errc::model_execution_failed,
                                  describe_error(sampled_tokens.error()) });
    }

    const auto sample_count = batch->sample_count();
    if (sampled_tokens->size() != sample_count) {
        return fail_after_abort(*scheduled,
                                { inference_engine_errc::runner_result_count_mismatch,
                                  "expected "
                                      + std::to_string(sample_count)
                                      + " samples, received "
                                      + std::to_string(sampled_tokens->size()) });
    }

    auto completed = scheduler_.complete(scheduled->id, *sampled_tokens);
    if (!completed) {
        return fail_after_abort(
            *scheduled,
            { inference_engine_errc::batch_completion_failed, describe_error(completed.error()) });
    }
    return std::move(*completed);
}

result<void, inference_engine_error>
inference_engine::cancel(seq_id id)
{
    auto cancelled = scheduler_.cancel(id);
    if (!cancelled) {
        return fail(inference_engine_errc::sequence_cancel_failed, cancelled.error());
    }
    return {};
}

result<void, inference_engine_error>
inference_engine::remove(seq_id id)
{
    auto removed = scheduler_.remove(id);
    if (!removed) {
        return fail(inference_engine_errc::sequence_remove_failed, removed.error());
    }
    return {};
}

std::unexpected<inference_engine_error>
inference_engine::fail_after_abort(const scheduled_batch& batch, inference_engine_error error)
{
    auto aborted = scheduler_.abort(batch.id);
    if (!aborted) {
        return fail(inference_engine_error {
            inference_engine_errc::batch_abort_failed,
            describe_error(aborted.error()) + "; original failure: " + describe_error(error) });
    }

    return fail(std::move(error));
}

} // namespace chibillm
