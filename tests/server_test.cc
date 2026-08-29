#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <condition_variable>
#include <future>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "server.h"

namespace {

using chibillm::chat_message;
using chibillm::model_batch;
using chibillm::model_info;
using chibillm::model_runner;
using chibillm::model_runner_errc;
using chibillm::result;
using chibillm::token_id;
using json = nlohmann::json;

class recording_runner final : public model_runner {
public:
    const model_info&
    info() const noexcept override
    {
        return info_;
    }

    result<std::vector<token_id>, model_runner_errc>
    encode_chat(std::span<const chat_message> messages) override
    {
        if (messages.empty()) {
            return chibillm::fail(model_runner_errc::invalid_chat);
        }
        return std::vector<token_id> { 1 };
    }

    result<std::string, model_runner_errc>
    decode(std::span<const token_id> tokens) const override
    {
        return std::string(tokens.size(), '*');
    }

    result<std::vector<token_id>, model_runner_errc>
    execute(const model_batch& batch) override
    {
        {
            std::unique_lock lock(mutex_);
            largest_batch_ = std::max(largest_batch_, batch.sequence_count());
            if (!first_execute_) {
                first_execute_ = true;
                changed_.notify_all();
                changed_.wait(lock, [&] { return released_; });
            }
        }
        return std::vector<token_id>(batch.sequence_count(), '*');
    }

    bool
    wait_for_first_execute()
    {
        std::unique_lock lock(mutex_);
        return changed_.wait_for(lock, std::chrono::seconds(2), [&] { return first_execute_; });
    }

    void
    release()
    {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        changed_.notify_all();
    }

    std::size_t
    largest_batch() const
    {
        std::lock_guard lock(mutex_);
        return largest_batch_;
    }

private:
    model_info info_ { .id = "test-model", .max_context_tokens = 64, .eos_token = 0 };
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::size_t largest_batch_ {};
    bool first_execute_ {};
    bool released_ {};
};

std::string
request_body(bool stream = false)
{
    return json({ { "model", "test-model" },
                  { "messages", json::array({ { { "role", "user" }, { "content", "hi" } } }) },
                  { "max_completion_tokens", 2 },
                  { "stream", stream } })
        .dump();
}

httplib::Result
post(std::uint16_t port, std::string body)
{
    httplib::Client client("127.0.0.1", port);
    return client.Post("/v1/chat/completions", body, "application/json");
}

} // namespace

TEST_CASE("OpenAI server batches concurrent requests and returns compatible chat responses")
{
    recording_runner runner;
    auto server = chibillm::openai_server::make(runner,
                                                {
                                                    .host = "127.0.0.1",
                                                    .port = 0,
                                                    .max_sequences = 2,
                                                    .max_pending_requests = 8,
                                                    .max_batch_tokens = 8,
                                                    .kv_block_count = 8,
                                                    .kv_block_size = 2,
                                                    .default_max_completion_tokens = 2,
                                                });
    REQUIRE(server.has_value());

    std::optional<chibillm::result<void, chibillm::server_errc>> run_result;
    std::jthread server_thread([&] { run_result = (*server)->run(); });

    auto first =
        std::async(std::launch::async, [&] { return post((*server)->port(), request_body()); });
    const bool started = runner.wait_for_first_execute();
    auto second =
        std::async(std::launch::async, [&] { return post((*server)->port(), request_body()); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    runner.release();

    auto first_response = first.get();
    auto second_response = second.get();
    auto stream_response = post((*server)->port(), request_body(true));

    httplib::Client client("127.0.0.1", (*server)->port());
    auto models_response = client.Get("/v1/models");
    auto missing_response =
        post((*server)->port(),
             json({ { "model", "missing" },
                    { "messages", json::array({ { { "role", "user" }, { "content", "hi" } } }) } })
                 .dump());

    (*server)->stop();
    server_thread.join();

    REQUIRE(started);
    REQUIRE(first_response);
    REQUIRE(second_response);
    CHECK(first_response->status == 200);
    CHECK(second_response->status == 200);
    const auto completion = json::parse(first_response->body);
    CHECK(completion["object"] == "chat.completion");
    CHECK(completion["choices"][0]["message"]["content"] == "**");
    CHECK(completion["choices"][0]["finish_reason"] == "length");
    CHECK(completion["usage"]["total_tokens"] == 3);
    CHECK(runner.largest_batch() == 2);

    REQUIRE(stream_response);
    CHECK(stream_response->status == 200);
    CHECK(stream_response->body.find("\"object\":\"chat.completion.chunk\"") != std::string::npos);
    CHECK(stream_response->body.find("\"delta\":{},\"finish_reason\":\"length\"")
          != std::string::npos);
    CHECK(stream_response->body.ends_with("data: [DONE]\n\n"));

    REQUIRE(models_response);
    CHECK(json::parse(models_response->body)["data"][0]["id"] == "test-model");
    REQUIRE(missing_response);
    CHECK(missing_response->status == 404);
    CHECK(json::parse(missing_response->body)["error"]["code"] == "model_not_found");
    REQUIRE(run_result.has_value());
    CHECK(run_result->has_value());
}
