#include "serving_runtime.h"
#include <chrono>
#include <deque>
#include <thread>
#include <unordered_map>

namespace chibillm {
struct prepared_request {
    std::shared_ptr<request_state> state;
    std::vector<token_id> prompt;
    std::size_t max_completion_tokens;
};

struct serving_runtime::implementation {
public:
    static result<std::unique_ptr<implementation>, inference_engine_errc>
    make(model_runner& runner, const serving_config& config)
    {
        auto engine = inference_engine::make(
            {
                .max_sequences = config.max_sequences,
                .max_batch_tokens = config.max_batch_tokens,
                .kv_block_count = config.kv_block_count,
                .kv_block_size = config.kv_block_size,
                .eos_token = runner.info().eos_token,
            },
            runner);
        if (!engine) {
            return fail(inference_engine_errc::scheduler_creation_failed);
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
    submit(generation_request request)
    {
        auto state =
            std::make_shared<request_state>(next_id_.fetch_add(1, std::memory_order_relaxed),
                                            std::chrono::duration_cast<std::chrono::seconds>(
                                                std::chrono::system_clock::now().time_since_epoch())
                                                .count());
        {
            std::lock_guard lock(mutex_);
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

        std::unique_lock lock(state->mutex);
        state->changed.wait(lock, [&] { return state->status != request_status::submitted; });
        if (state->error) {
            return fail(*state->error);
        }
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
        std::deque<submission> submissions;
        {
            std::lock_guard lock(mutex_);
            submissions.swap(submissions_);
        }

        for (auto& submission : submissions) {
            auto prompt = runner_.encode_chat(submission.request.messages);
            if (!prompt) {
                finish_error(submission.state,
                             { .kind = generation_errc::invalid_input,
                               .message = "The messages could not be encoded.",
                               .param = "messages" });
                continue;
            }
            if (prompt->size() >= runner_.info().max_context_tokens
                || submission.request.max_completion_tokens
                    > runner_.info().max_context_tokens - prompt->size()) {
                finish_error(
                    submission.state,
                    { .kind = generation_errc::invalid_input,
                      .message =
                          "The requested prompt and completion exceed the model context window.",
                      .param = "max_completion_tokens",
                      .code = "context_length_exceeded" });
                continue;
            }

            {
                std::lock_guard lock(submission.state->mutex);
                submission.state->decoder = runner_.make_decoder();
                submission.state->prompt_tokens = prompt->size();
                submission.state->status = request_status::ready;
            }
            submission.state->changed.notify_all();
            pending_.push_back(
                { submission.state, std::move(*prompt), submission.request.max_completion_tokens });
        }
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
            if (!engine_.cancel(id) || !engine_.remove(id)) {
                finish_error(active->second,
                             {
                                 .kind = generation_errc::execution_failure,
                                 .message = "The request could not be cancelled.",
                             });
            } else {
                finish_cancelled(active->second);
            }
            active = active_.erase(active);
        }
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
        while (active_.size() < config_.max_sequences && !pending_.empty()) {
            auto request = std::move(pending_.front());
            pending_.pop_front();
            auto sequence = seq::make(request.state->id, std::move(request.prompt),
                                      {
                                          .max_new_tokens = request.max_completion_tokens,
                                          .ignore_eos = false,
                                      });
            if (!sequence || !engine_.add(std::move(*sequence))) {
                finish_error(request.state,
                             {
                                 .kind = generation_errc::execution_failure,
                                 .message = "The request could not be admitted.",
                             });
                continue;
            }
            active_.emplace(request.state->id, std::move(request.state));
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
            [[maybe_unused]] const auto cancelled = engine_.cancel(update.id);
            finish_error(state,
                         { .kind = generation_errc::execution_failure,
                           .message = "The generated text could not be decoded." });
            return true;
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
            fail_all("Model execution failed.");
            return;
        }
        for (const auto& update : *updates) {
            if (!append_update(update)) {
                continue;
            }
            [[maybe_unused]] const auto removed = engine_.remove(update.id);
            active_.erase(update.id);
        }
    }

    void
    fail_all(std::string message)
    {
        for (auto& request : pending_) {
            finish_error(request.state,
                         {
                             .kind = generation_errc::execution_failure,
                             .message = message,
                         });
        }
        pending_.clear();
        for (auto& [id, state] : active_) {
            [[maybe_unused]] const auto cancelled = engine_.cancel(id);
            [[maybe_unused]] const auto removed = engine_.remove(id);
            finish_error(state,
                         {
                             .kind = generation_errc::execution_failure,
                             .message = message,
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
        drain_submissions();
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

result<std::unique_ptr<serving_runtime>, inference_engine_errc>
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
    return implementation_->submit(std::move(request));
}

void
serving_runtime::cancel(const std::shared_ptr<request_state>& state) noexcept
{
    implementation_->cancel(state);
}
} // namespace chibillm
