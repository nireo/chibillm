#include "serving_runtime.h"
#include <algorithm>
#include <chrono>
#include <deque>
#include <thread>
#include <unordered_map>

namespace chibillm {
struct prepared_request {
    std::shared_ptr<request_state> state;
    seq sequence;
};

struct serving_runtime::implementation {
public:
    static result<std::unique_ptr<implementation>, inference_engine_error>
    make(model_runner& runner, const serving_config& config)
    {
        auto engine = inference_engine::make(config.scheduler, runner);
        if (!engine) {
            return fail(engine.error());
        }
        return std::unique_ptr<implementation>(
            new implementation(runner, config, std::move(*engine)));
    }

    ~implementation()
    {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        changed_.notify_one();
        worker_.request_stop();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    result<std::shared_ptr<request_state>, generation_error>
    enqueue(generation_request request)
    {
        auto state =
            std::make_shared<request_state>(next_id_.fetch_add(1, std::memory_order_relaxed),
                                            std::chrono::duration_cast<std::chrono::seconds>(
                                                std::chrono::system_clock::now().time_since_epoch())
                                                .count());
        {
            std::lock_guard lock(mutex_);
            if (stopping_)
                return fail(generation_error { .kind = generation_errc::execution_failure,
                                               .message = "The server is shutting down." });
            if (outstanding_ >= config_.max_pending_requests) {
                return fail(generation_error {
                    .kind = generation_errc::queue_full,
                    .message = "The server request queue is full.",
                    .code = "queue_full",
                });
            }
            ++outstanding_;
            submissions_.push_back({ std::move(request), state });
        }
        changed_.notify_one();
        return state;
    }

    void
    cancel(const std::shared_ptr<request_state>& state) noexcept
    {
        state->cancelled.store(true, std::memory_order_relaxed);
        changed_.notify_one();
    }

private:
    struct submission {
        generation_request request;
        std::shared_ptr<request_state> state;
    };

    implementation(model_runner& runner, serving_config config, inference_engine engine)
        : runner_(runner)
        , config_(std::move(config))
        , engine_(std::move(engine))
        , worker_([this](std::stop_token stop) { run(stop); })
    {}

    void
    finish_error(const std::shared_ptr<request_state>& state, generation_error error)
    {
        {
            std::lock_guard lock(state->mutex);
            state->error = std::move(error);
            state->status = request_status::finished;
        }
        state->changed.notify_all();
        complete_one();
    }

    void
    complete_one()
    {
        std::lock_guard lock(mutex_);
        --outstanding_;
    }

    void
    drain_submissions()
    {
        // Bound preparation between model steps; do not tokenize a whole burst at once.
        std::optional<submission> next;
        {
            std::lock_guard lock(mutex_);
            if (submissions_.empty())
                return;
            next.emplace(std::move(submissions_.front()));
            submissions_.pop_front();
        }

        auto& submission = *next;
        const auto context_limit =
            std::min(runner_.info().max_context_tokens,
                     config_.max_context_tokens.value_or(runner_.info().max_context_tokens));
        if (submission.state->cancelled.load(std::memory_order_relaxed)) {
            finish_cancelled(submission.state);
            return;
        }
        auto prompt = runner_.encode_chat(submission.request.messages);
        if (!prompt) {
            finish_error(
                submission.state,
                { .kind = generation_errc::invalid_input,
                  .message = "The messages could not be encoded: " + describe_error(prompt.error()),
                  .param = "messages",
                  .code = std::string(error_name(prompt.error().code)) });
            return;
        }
        if (prompt->tokens.size() >= context_limit
            || submission.request.max_completion_tokens > context_limit - prompt->tokens.size()) {
            finish_error(
                submission.state,
                { .kind = generation_errc::invalid_input,
                  .message = "The requested prompt and completion exceed the model context window.",
                  .param = "max_completion_tokens",
                  .code = "context_length_exceeded" });
            return;
        }

        auto sequence = seq::make(submission.state->id, std::move(prompt->tokens),
                                  { .max_new_tokens = submission.request.max_completion_tokens },
                                  std::move(prompt->checkpoints));
        if (!sequence) {
            finish_error(
                submission.state,
                { .kind = generation_errc::invalid_input,
                  .message = "Sequence creation failed: " + describe_error(sequence.error()),
                  .param = "max_completion_tokens" });
            return;
        }
        if (!engine_.fits_cache(*sequence)) {
            finish_error(
                submission.state,
                { .kind = generation_errc::invalid_input,
                  .message = "The prompt and completion cannot fit in the server KV cache.",
                  .param = "max_completion_tokens",
                  .code = "cache_capacity_exceeded" });
            return;
        }

        auto decoder = runner_.make_decoder();
        if (!decoder) {
            finish_error(submission.state,
                         { .kind = generation_errc::execution_failure,
                           .message = "Decoder setup failed: decoder creation returned null." });
            return;
        }
        {
            std::lock_guard lock(submission.state->mutex);
            submission.state->decoder = std::move(decoder);
            submission.state->prompt_tokens = sequence->prompt_token_count();
            submission.state->status = request_status::ready;
        }
        submission.state->changed.notify_all();
        pending_.push_back({ submission.state, std::move(*sequence) });
    }

    void
    cancel_requests()
    {
        for (auto pending = pending_.begin(); pending != pending_.end();) {
            if (!pending->state->cancelled.load(std::memory_order_relaxed)) {
                ++pending;
                continue;
            }
            finish_cancelled(pending->state);
            pending = pending_.erase(pending);
        }

        for (auto active = active_.begin(); active != active_.end();) {
            if (!active->second->cancelled.load(std::memory_order_relaxed)) {
                ++active;
                continue;
            }
            const auto id = active->first;
            auto released = release_sequence(id);
            if (!released) {
                finish_error(active->second,
                             {
                                 .kind = generation_errc::execution_failure,
                                 .message = "The request could not be cancelled: "
                                     + describe_error(released.error()),
                             });
            } else {
                finish_cancelled(active->second);
            }
            active = active_.erase(active);
        }
    }

    result<void, inference_engine_error>
    release_sequence(seq_id id)
    {
        auto cancelled = engine_.cancel(id);
        if (!cancelled)
            return fail(cancelled.error());
        return engine_.remove(id);
    }

    void
    finish_cancelled(const std::shared_ptr<request_state>& state)
    {
        {
            std::lock_guard lock(state->mutex);
            state->reason = finish_reason::cancelled;
            state->status = request_status::finished;
        }
        state->changed.notify_all();
        complete_one();
    }

    void
    admit_requests()
    {
        while (!pending_.empty()) {
            auto& request = pending_.front();
            auto added = engine_.try_add(request.sequence);
            if (added && *added == admission_result::deferred)
                break; // FIFO: later requests do not bypass the oldest pending request.
            if (!added) {
                finish_error(request.state,
                             {
                                 .kind = generation_errc::execution_failure,
                                 .message = "The request could not be admitted: "
                                     + describe_error(added.error()),
                             });
            } else if (*added == admission_result::too_large) {
                finish_error(request.state,
                             { .kind = generation_errc::invalid_input,
                               .message = "The request exceeds the server KV cache capacity.",
                               .param = "max_completion_tokens",
                               .code = "cache_capacity_exceeded" });
            } else {
                {
                    std::lock_guard lock(request.state->mutex);
                    request.state->cached_prompt_tokens =
                        engine_.find_sequence(request.state->id)->cached_token_count();
                }
                active_.emplace(request.state->id, request.state);
            }
            pending_.pop_front();
        }
    }

    bool
    append_update(const sequence_update& update)
    {
        const auto found = active_.find(update.id);
        if (found == active_.end()) {
            return false;
        }
        const auto& state = found->second;
        const bool finished = update.reason != finish_reason::none;
        auto delta = state->decoder->push(update.token, finished);
        if (!delta) {
            auto message =
                "The generated text could not be decoded: " + describe_error(delta.error());
            auto released = release_sequence(update.id);
            if (!released)
                message += "; cleanup failed: " + describe_error(released.error());
            finish_error(state,
                         { .kind = generation_errc::execution_failure,
                           .message = std::move(message),
                           .code = std::string(error_name(delta.error().code)) });
            return true;
        }
        if (finished) {
            auto removed = engine_.remove(update.id);
            if (!removed) {
                finish_error(
                    state,
                    { .kind = generation_errc::execution_failure,
                      .message = "Sequence cleanup failed: " + describe_error(removed.error()) });
                return true;
            }
        }
        {
            std::lock_guard lock(state->mutex);
            state->output += *delta;
            ++state->completion_tokens;
            if (finished) {
                state->reason = update.reason;
                state->status = request_status::finished;
            }
        }
        state->changed.notify_all();
        if (finished)
            complete_one();
        return finished;
    }

    void
    execute_step()
    {
        auto updates = engine_.step();
        if (!updates) {
            fail_all("Model execution failed: " + describe_error(updates.error()),
                     std::string(error_name(updates.error().code)));
            return;
        }
        for (const auto& update : *updates) {
            if (!append_update(update)) {
                continue;
            }
            active_.erase(update.id);
        }
    }

    void
    fail_all(std::string message, std::string code = {})
    {
        for (auto& request : pending_) {
            finish_error(request.state,
                         {
                             .kind = generation_errc::execution_failure,
                             .message = message,
                             .code = code,
                         });
        }
        pending_.clear();
        for (auto& [id, state] : active_) {
            auto request_message = message;
            auto released = release_sequence(id);
            if (!released)
                request_message += "; cleanup failed: " + describe_error(released.error());
            finish_error(state,
                         {
                             .kind = generation_errc::execution_failure,
                             .message = std::move(request_message),
                             .code = code,
                         });
        }
        active_.clear();
    }

    void
    run(std::stop_token stop)
    {
        while (!stop.stop_requested()) {
            drain_submissions();
            cancel_requests();
            admit_requests();
            if (!active_.empty()) {
                execute_step();
                continue;
            }

            std::unique_lock lock(mutex_);
            changed_.wait(lock, [&] { return stopping_ || !submissions_.empty(); });
            if (stopping_) {
                break;
            }
        }
        std::deque<submission> submissions;
        {
            std::lock_guard lock(mutex_);
            submissions.swap(submissions_);
        }
        for (const auto& submission : submissions)
            finish_error(submission.state,
                         { .kind = generation_errc::execution_failure,
                           .message = "The server is shutting down." });
        fail_all("The server is shutting down.");
    }

    model_runner& runner_;
    serving_config config_;
    inference_engine engine_;
    std::deque<prepared_request> pending_;
    std::unordered_map<seq_id, std::shared_ptr<request_state>> active_;

    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<submission> submissions_;
    std::size_t outstanding_ {};
    bool stopping_ {};
    std::atomic<seq_id> next_id_ { 1 };
    std::jthread worker_;
};

result<std::unique_ptr<serving_runtime>, inference_engine_error>
serving_runtime::make(model_runner& runner, serving_config config)
{
    auto impl = implementation::make(runner, config);
    if (!impl)
        return fail(impl.error());
    return std::unique_ptr<serving_runtime>(new serving_runtime(std::move(*impl)));
}

serving_runtime::serving_runtime(std::unique_ptr<implementation> impl)
    : implementation_(std::move(impl))
{}

serving_runtime::~serving_runtime() = default;

result<std::shared_ptr<request_state>, generation_error>
serving_runtime::submit(generation_request request)
{
    auto submitted = enqueue(std::move(request));
    if (!submitted)
        return fail(submitted.error());
    auto state = *submitted;
    std::unique_lock lock(state->mutex);
    state->changed.wait(lock, [&] { return state->status != request_status::submitted; });
    if (state->error)
        return fail(*state->error);
    return state;
}

result<std::shared_ptr<request_state>, generation_error>
serving_runtime::enqueue(generation_request request)
{
    return implementation_->enqueue(std::move(request));
}

void
serving_runtime::cancel(const std::shared_ptr<request_state>& state) noexcept
{
    implementation_->cancel(state);
}
} // namespace chibillm
