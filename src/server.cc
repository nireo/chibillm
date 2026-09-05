#include "server.h"
#include "serving_runtime.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
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
    generation_request generation;
    bool stream;
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
        .generation = { .messages = {}, .max_completion_tokens = default_max_tokens },
        .stream = false,
    };
    request.generation.messages.reserve(input["messages"].size());
    for (const auto& value : input["messages"]) {
        if (!value.is_object()
            || !value.contains("role")
            || !value["role"].is_string()
            || !value.contains("content")
            || !value["content"].is_string()) {
            return fail(
                invalid("Each message must have string role and content fields.", "messages"));
        }
        request.generation.messages.push_back(
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
        request.generation.max_completion_tokens = static_cast<std::size_t>(amount);
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

request_error
http_error(const generation_error& error)
{
    const auto status = error.kind == generation_errc::queue_full ? 429
        : error.kind == generation_errc::invalid_input            ? 400
                                                                  : 500;
    return { .status = status,
             .message = error.message,
             .type = status == 500 ? "server_error"
                 : status == 429   ? "rate_limit_error"
                                   : "invalid_request_error",
             .param = error.param,
             .code = error.code };
}

json
error_json(const generation_error& error)
{
    return error_json(http_error(error));
}

void
write_error(httplib::Response& response, const generation_error& error)
{
    write_error(response, http_error(error));
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
    auto runtime = serving_runtime::make(runner,
                                         { .max_sequences = config.max_sequences,
                                           .max_pending_requests = config.max_pending_requests,
                                           .max_batch_tokens = config.max_batch_tokens,
                                           .kv_block_count = config.kv_block_count,
                                           .kv_block_size = config.kv_block_size });
    if (!runtime) {
        return fail(server_errc::engine_creation_failed);
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
            auto submitted = runtime_ptr->submit(std::move(parsed->generation));
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
                    std::optional<generation_error> error;
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
