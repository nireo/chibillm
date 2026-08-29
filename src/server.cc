#include "server.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "inference_engine.h"

namespace chibillm {
namespace {

using json = nlohmann::json;

struct request_error {
    int status;
    std::string message;
    std::string type { "invalid_request_error" };
    std::string param;
    std::string code;
};

struct completion_request {
    std::vector<chat_message> messages;
    std::size_t max_completion_tokens;
    bool stream;
};

enum class request_status : std::uint8_t {
    submitted,
    ready,
    finished,
};

struct request_state {
    request_state(seq_id id, std::int64_t created) noexcept
        : id(id)
        , created(created)
    {}

    seq_id id;
    std::int64_t created;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::string output;
    std::string pending_bytes;
    std::size_t prompt_tokens {};
    std::size_t completion_tokens {};
    finish_reason reason { finish_reason::none };
    std::optional<request_error> error;
    request_status status { request_status::submitted };
    std::atomic_bool cancelled {};
};

struct prepared_request {
    std::shared_ptr<request_state> state;
    std::vector<token_id> prompt;
    std::size_t max_completion_tokens;
};

[[nodiscard]] std::optional<std::size_t>
complete_utf8_prefix(std::string_view text)
{
    std::size_t offset = 0;
    while (offset < text.size()) {
        const auto first = static_cast<unsigned char>(text[offset]);
        const std::size_t length = first < 0x80 ? 1
            : (first & 0xE0) == 0xC0            ? 2
            : (first & 0xF0) == 0xE0            ? 3
            : (first & 0xF8) == 0xF0            ? 4
                                                : 0;
        if (length == 0) {
            return std::nullopt;
        }
        if (length > text.size() - offset) {
            break;
        }
        for (std::size_t index = 1; index < length; ++index) {
            if ((static_cast<unsigned char>(text[offset + index]) & 0xC0) != 0x80) {
                return std::nullopt;
            }
        }
        offset += length;
    }
    return offset;
}

class serving_runtime {
public:
    static result<std::unique_ptr<serving_runtime>, server_errc>
    make(model_runner& runner, const server_config& config)
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
            return fail(server_errc::engine_creation_failed);
        }
        return std::unique_ptr<serving_runtime>(
            new serving_runtime(runner, config, std::move(*engine)));
    }

    ~serving_runtime()
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

    result<std::shared_ptr<request_state>, request_error>
    submit(completion_request request)
    {
        auto state =
            std::make_shared<request_state>(next_id_.fetch_add(1, std::memory_order_relaxed),
                                            std::chrono::duration_cast<std::chrono::seconds>(
                                                std::chrono::system_clock::now().time_since_epoch())
                                                .count());
        {
            std::lock_guard lock(mutex_);
            if (outstanding_ >= config_.max_pending_requests) {
                return fail(request_error {
                    .status = 429,
                    .message = "The server request queue is full.",
                    .type = "rate_limit_error",
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
        completion_request request;
        std::shared_ptr<request_state> state;
    };

    serving_runtime(model_runner& runner, server_config config, inference_engine engine)
        : runner_(runner)
        , config_(std::move(config))
        , engine_(std::move(engine))
        , worker_([this](std::stop_token stop) { run(stop); })
    {}

    void
    finish_error(const std::shared_ptr<request_state>& state, request_error error)
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
                             { .status = 400,
                               .message = "The messages could not be encoded.",
                               .param = "messages" });
                continue;
            }
            if (prompt->size() >= runner_.info().max_context_tokens
                || submission.request.max_completion_tokens
                    > runner_.info().max_context_tokens - prompt->size()) {
                finish_error(
                    submission.state,
                    { .status = 400,
                      .message =
                          "The requested prompt and completion exceed the model context window.",
                      .param = "max_completion_tokens",
                      .code = "context_length_exceeded" });
                continue;
            }

            {
                std::lock_guard lock(submission.state->mutex);
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
                             { .status = 500,
                               .message = "The request could not be cancelled.",
                               .type = "server_error" });
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
                                          .temperature = 1.0F,
                                          .max_new_tokens = request.max_completion_tokens,
                                          .ignore_eos = false,
                                      },
                                      config_.kv_block_size);
            if (!sequence || !engine_.add(std::move(*sequence))) {
                finish_error(request.state,
                             { .status = 500,
                               .message = "The request could not be admitted.",
                               .type = "server_error" });
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
        const std::span token(&update.token, 1);
        auto bytes = runner_.decode(token);
        if (!bytes) {
            [[maybe_unused]] const auto cancelled = engine_.cancel(update.id);
            finish_error(state,
                         { .status = 500,
                           .message = "The generated token could not be decoded.",
                           .type = "server_error" });
            return true;
        }

        bool finished = update.reason != finish_reason::none;
        bool invalid_text = false;
        {
            std::lock_guard lock(state->mutex);
            state->pending_bytes += *bytes;
            ++state->completion_tokens;
            const auto prefix = complete_utf8_prefix(state->pending_bytes);
            if (!prefix) {
                invalid_text = true;
            } else if (*prefix != 0) {
                state->output.append(state->pending_bytes, 0, *prefix);
                state->pending_bytes.erase(0, *prefix);
            }
            if (finished && !state->pending_bytes.empty()) {
                invalid_text = true;
            }
            if (!invalid_text && finished) {
                state->reason = update.reason;
                state->status = request_status::finished;
            }
        }

        if (invalid_text) {
            [[maybe_unused]] const auto cancelled = engine_.cancel(update.id);
            finish_error(state,
                         { .status = 500,
                           .message = "The model produced invalid UTF-8.",
                           .type = "server_error" });
        } else {
            state->changed.notify_all();
            if (finished) {
                complete_one();
            }
        }
        return finished || invalid_text;
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
                         { .status = 500, .message = message, .type = "server_error" });
        }
        pending_.clear();
        for (auto& [id, state] : active_) {
            [[maybe_unused]] const auto cancelled = engine_.cancel(id);
            [[maybe_unused]] const auto removed = engine_.remove(id);
            finish_error(state, { .status = 500, .message = message, .type = "server_error" });
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
    server_config config_;
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

[[nodiscard]] request_error
invalid(std::string message, std::string param = {})
{
    return { .status = 400, .message = std::move(message), .param = std::move(param) };
}

result<completion_request, request_error>
parse_completion_request(std::string_view body,
                         const model_info& model,
                         std::size_t default_max_tokens)
{
    auto input = json::parse(body, nullptr, false);
    if (input.is_discarded() || !input.is_object()) {
        return fail(invalid("The request body must be a JSON object."));
    }
    if (!input.contains("model") || !input["model"].is_string()) {
        return fail(invalid("The model field is required.", "model"));
    }
    if (input["model"].get_ref<const std::string&>() != model.id) {
        return fail(request_error {
            .status = 404,
            .message = "The requested model does not exist.",
            .param = "model",
            .code = "model_not_found",
        });
    }
    if (!input.contains("messages") || !input["messages"].is_array() || input["messages"].empty()) {
        return fail(invalid("The messages field must be a non-empty array.", "messages"));
    }

    completion_request request {
        .messages = {},
        .max_completion_tokens = default_max_tokens,
        .stream = false,
    };
    request.messages.reserve(input["messages"].size());
    for (const auto& value : input["messages"]) {
        if (!value.is_object()
            || !value.contains("role")
            || !value["role"].is_string()
            || !value.contains("content")
            || !value["content"].is_string()) {
            return fail(
                invalid("Each message must have string role and content fields.", "messages"));
        }
        request.messages.push_back(
            { value["role"].get<std::string>(), value["content"].get<std::string>() });
    }

    const auto max_field =
        input.contains("max_completion_tokens") && !input["max_completion_tokens"].is_null()
        ? "max_completion_tokens"
        : input.contains("max_tokens") && !input["max_tokens"].is_null() ? "max_tokens"
                                                                         : nullptr;
    if (max_field != nullptr) {
        const auto& value = input[max_field];
        if (!value.is_number_integer()) {
            return fail(invalid("The maximum completion token count must be a positive integer.",
                                max_field));
        }
        std::uint64_t amount = 0;
        if (value.is_number_unsigned()) {
            amount = value.get<std::uint64_t>();
        } else {
            const auto signed_amount = value.get<std::int64_t>();
            if (signed_amount > 0) {
                amount = static_cast<std::uint64_t>(signed_amount);
            }
        }
        if (amount == 0 || amount > std::numeric_limits<std::size_t>::max()) {
            return fail(invalid("The maximum completion token count must be a positive integer.",
                                max_field));
        }
        request.max_completion_tokens = static_cast<std::size_t>(amount);
    }
    if (input.contains("stream") && !input["stream"].is_null()) {
        if (!input["stream"].is_boolean()) {
            return fail(invalid("The stream field must be boolean.", "stream"));
        }
        request.stream = input["stream"].get<bool>();
    }
    if (input.contains("n")
        && !input["n"].is_null()
        && (!input["n"].is_number_integer() || input["n"] != 1)) {
        return fail(invalid("Only n=1 is supported.", "n"));
    }
    if ((input.contains("tools") && !input["tools"].is_null())
        || (input.contains("tool_choice") && !input["tool_choice"].is_null())) {
        return fail(invalid("Tools are not supported by this model.", "tools"));
    }
    if (input.contains("stop") && !input["stop"].is_null()) {
        return fail(invalid("Custom stop sequences are not supported.", "stop"));
    }
    if (input.contains("temperature")
        && !input["temperature"].is_null()
        && (!input["temperature"].is_number() || input["temperature"] != 0)) {
        return fail(
            invalid("Only greedy sampling with temperature=0 is supported.", "temperature"));
    }
    return request;
}

json
error_json(const request_error& error)
{
    return { { "error",
               { { "message", error.message },
                 { "type", error.type },
                 { "param", error.param.empty() ? json(nullptr) : json(error.param) },
                 { "code", error.code.empty() ? json(nullptr) : json(error.code) } } } };
}

void
write_error(httplib::Response& response, const request_error& error)
{
    response.status = error.status;
    response.set_content(error_json(error).dump(), "application/json");
}

std::string
completion_id(seq_id id)
{
    return "chatcmpl-" + std::to_string(id);
}

std::string
openai_finish_reason(finish_reason reason)
{
    return reason == finish_reason::len_limit ? "length" : "stop";
}

json
usage_json(const request_state& state)
{
    return {
        { "prompt_tokens", state.prompt_tokens },
        { "completion_tokens", state.completion_tokens },
        { "total_tokens", state.prompt_tokens + state.completion_tokens },
    };
}

json
completion_json(const request_state& state, const model_info& model)
{
    return {
        { "id", completion_id(state.id) },
        { "object", "chat.completion" },
        { "created", state.created },
        { "model", model.id },
        { "choices",
          json::array({ { { "index", 0 },
                          { "message",
                            { { "role", "assistant" },
                              { "content", state.output },
                              { "refusal", nullptr } } },
                          { "logprobs", nullptr },
                          { "finish_reason", openai_finish_reason(state.reason) } } }) },
        { "usage", usage_json(state) },
    };
}

json
chunk_json(const request_state& state,
           const model_info& model,
           json delta,
           json finish_reason_value)
{
    return {
        { "id", completion_id(state.id) },
        { "object", "chat.completion.chunk" },
        { "created", state.created },
        { "model", model.id },
        { "choices",
          json::array({ { { "index", 0 },
                          { "delta", std::move(delta) },
                          { "logprobs", nullptr },
                          { "finish_reason", std::move(finish_reason_value) } } }) },
    };
}

void
write_sse(httplib::DataSink& sink, const json& value)
{
    const auto data = "data: " + value.dump() + "\n\n";
    sink.write(data.data(), data.size());
}

} // namespace

struct openai_server::implementation {
    implementation(model_runner& runner,
                   server_config config,
                   std::unique_ptr<serving_runtime> runtime)
        : runner(runner)
        , config(std::move(config))
        , runtime(std::move(runtime))
    {}

    model_runner& runner;
    server_config config;
    std::unique_ptr<serving_runtime> runtime;
    httplib::Server http;
    std::uint16_t bound_port {};
};

result<std::unique_ptr<openai_server>, server_errc>
openai_server::make(model_runner& runner, server_config config)
{
    if (config.host.empty()
        || config.max_sequences == 0
        || config.max_pending_requests == 0
        || config.max_batch_tokens == 0
        || config.kv_block_count == 0
        || config.kv_block_size == 0
        || config.default_max_completion_tokens == 0) {
        return fail(server_errc::invalid_config);
    }
    auto runtime = serving_runtime::make(runner, config);
    if (!runtime) {
        return fail(runtime.error());
    }
    auto impl = std::make_unique<implementation>(runner, std::move(config), std::move(*runtime));

    impl->http.Get("/health", [](const httplib::Request&, httplib::Response& response) {
        response.set_content(R"({"status":"ok"})", "application/json");
    });
    impl->http.Get("/v1/models", [&runner](const httplib::Request&, httplib::Response& response) {
        const auto& model = runner.info();
        const json body {
            { "object", "list" },
            { "data",
              json::array({ { { "id", model.id },
                              { "object", "model" },
                              { "created", 0 },
                              { "owned_by", "chibillm" } } }) },
        };
        response.set_content(body.dump(), "application/json");
    });
    impl->http.Get(R"(/v1/models/(.+))",
                   [&runner](const httplib::Request& request, httplib::Response& response) {
                       const auto id = request.matches[1].str();
                       if (id != runner.info().id) {
                           write_error(response,
                                       { .status = 404,
                                         .message = "The requested model does not exist.",
                                         .param = "model",
                                         .code = "model_not_found" });
                           return;
                       }
                       response.set_content(json({ { "id", id },
                                                   { "object", "model" },
                                                   { "created", 0 },
                                                   { "owned_by", "chibillm" } })
                                                .dump(),
                                            "application/json");
                   });
    auto* runtime_ptr = impl->runtime.get();
    impl->http.Post(
        "/v1/chat/completions",
        [&runner, runtime_ptr, default_max = impl->config.default_max_completion_tokens](
            const httplib::Request& request, httplib::Response& response) {
            auto parsed = parse_completion_request(request.body, runner.info(), default_max);
            if (!parsed) {
                write_error(response, parsed.error());
                return;
            }
            const bool stream = parsed->stream;
            auto submitted = runtime_ptr->submit(std::move(*parsed));
            if (!submitted) {
                write_error(response, submitted.error());
                return;
            }
            auto state = std::move(*submitted);
            if (!stream) {
                std::unique_lock lock(state->mutex);
                state->changed.wait(lock,
                                    [&] { return state->status == request_status::finished; });
                if (state->error) {
                    write_error(response, *state->error);
                    return;
                }
                response.set_content(completion_json(*state, runner.info()).dump(),
                                     "application/json");
                return;
            }

            struct stream_cursor {
                std::size_t sent {};
                bool sent_role {};
                bool sent_finish {};
            };
            auto cursor = std::make_shared<stream_cursor>();
            response.set_header("Cache-Control", "no-cache");
            response.set_header("X-Accel-Buffering", "no");
            response.set_chunked_content_provider(
                "text/event-stream",
                [state, cursor, &runner](std::size_t, httplib::DataSink& sink) {
                    std::string delta;
                    std::optional<request_error> error;
                    finish_reason reason = finish_reason::none;
                    bool finished = false;
                    {
                        std::unique_lock lock(state->mutex);
                        state->changed.wait(lock, [&] {
                            return state->output.size() > cursor->sent
                                || state->status == request_status::finished;
                        });
                        delta = state->output.substr(cursor->sent);
                        cursor->sent = state->output.size();
                        error = state->error;
                        reason = state->reason;
                        finished = state->status == request_status::finished;
                    }

                    if (error) {
                        write_sse(sink, error_json(*error));
                        sink.write("data: [DONE]\n\n", 14);
                        sink.done();
                        return true;
                    }
                    if (!cursor->sent_role) {
                        write_sse(sink,
                                  chunk_json(*state, runner.info(), { { "role", "assistant" } },
                                             nullptr));
                        cursor->sent_role = true;
                    }
                    if (!delta.empty()) {
                        write_sse(
                            sink,
                            chunk_json(*state, runner.info(), { { "content", delta } }, nullptr));
                    }
                    if (finished && !cursor->sent_finish) {
                        write_sse(sink,
                                  chunk_json(*state, runner.info(), json::object(),
                                             openai_finish_reason(reason)));
                        sink.write("data: [DONE]\n\n", 14);
                        cursor->sent_finish = true;
                        sink.done();
                    }
                    return true;
                },
                [state, runtime_ptr](bool success) {
                    if (!success) {
                        runtime_ptr->cancel(state);
                    }
                });
        });

    int port = -1;
    if (impl->config.port == 0) {
        port = impl->http.bind_to_any_port(impl->config.host);
    } else if (impl->http.bind_to_port(impl->config.host, impl->config.port)) {
        port = impl->config.port;
    }
    if (port <= 0) {
        return fail(server_errc::bind_failed);
    }
    impl->bound_port = static_cast<std::uint16_t>(port);
    return std::unique_ptr<openai_server>(new openai_server(std::move(impl)));
}

openai_server::openai_server(std::unique_ptr<implementation> implementation) noexcept
    : implementation_(std::move(implementation))
{}

openai_server::~openai_server()
{
    stop();
}

std::uint16_t
openai_server::port() const noexcept
{
    return implementation_->bound_port;
}

result<void, server_errc>
openai_server::run()
{
    if (!implementation_->http.listen_after_bind()) {
        return fail(server_errc::listen_failed);
    }
    return {};
}

void
openai_server::stop() noexcept
{
    if (implementation_) {
        implementation_->http.stop();
    }
}

} // namespace chibillm
