#include "inference_engine.h"

#include <utility>

#include "model_batch.h"

namespace chibillm {

result<inference_engine, inference_engine_errc>
inference_engine::make(scheduler_config config, model_runner& runner)
{
    auto scheduler_result = scheduler::make(config);
    if (!scheduler_result) {
        return fail(inference_engine_errc::scheduler_creation_failed);
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

result<void, inference_engine_errc>
inference_engine::add(seq sequence)
{
    auto added = scheduler_.add(std::move(sequence));
    if (!added) {
        return fail(inference_engine_errc::sequence_add_failed);
    }

    return {};
}

result<std::vector<sequence_update>, inference_engine_errc>
inference_engine::step()
{
    auto scheduled = scheduler_.schedule();
    if (!scheduled) {
        return fail(inference_engine_errc::scheduling_failed);
    }

    auto batch = build_model_batch(*scheduled, scheduler_);
    if (!batch) {
        return fail_after_abort(*scheduled, inference_engine_errc::model_batch_build_failed);
    }

    auto sampled_tokens = runner_->execute(*batch);
    if (!sampled_tokens) {
        return fail_after_abort(*scheduled, inference_engine_errc::model_execution_failed);
    }

    if (sampled_tokens->size() != scheduled->items.size()) {
        return fail_after_abort(*scheduled, inference_engine_errc::runner_result_count_mismatch);
    }

    std::vector<std::size_t> completion_counts;
    completion_counts.reserve(scheduled->items.size());
    for (const auto& item : scheduled->items) {
        const auto* sequence = scheduler_.find_sequence(item.id);
        if (sequence == nullptr) {
            return fail_after_abort(*scheduled, inference_engine_errc::batch_completion_failed);
        }
        completion_counts.push_back(sequence->completion_token_count());
    }

    auto completed = scheduler_.complete(*scheduled, *sampled_tokens);
    if (!completed) {
        return fail(inference_engine_errc::batch_completion_failed);
    }

    std::vector<sequence_update> updates;
    updates.reserve(scheduled->items.size());
    for (std::size_t index = 0; index < scheduled->items.size(); ++index) {
        const auto* sequence = scheduler_.find_sequence(scheduled->items[index].id);
        if (sequence != nullptr && sequence->completion_token_count() > completion_counts[index]) {
            updates.push_back({
                .id = sequence->id(),
                .token = sequence->last_token(),
                .reason = sequence->reason(),
            });
        }
    }
    return updates;
}

result<void, inference_engine_errc>
inference_engine::cancel(seq_id id)
{
    if (!scheduler_.cancel(id)) {
        return fail(inference_engine_errc::sequence_cancel_failed);
    }
    return {};
}

result<void, inference_engine_errc>
inference_engine::remove(seq_id id)
{
    if (!scheduler_.remove(id)) {
        return fail(inference_engine_errc::sequence_remove_failed);
    }
    return {};
}

std::unexpected<inference_engine_errc>
inference_engine::fail_after_abort(const scheduled_batch& batch, inference_engine_errc error)
{
    auto aborted = scheduler_.abort(batch);
    if (!aborted) {
        return fail(inference_engine_errc::batch_abort_failed);
    }

    return fail(error);
}

} // namespace chibillm
