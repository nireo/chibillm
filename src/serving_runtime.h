#pragma once
#include "inference_engine.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>

namespace chibillm {
struct serving_config {
    std::size_t max_sequences { 4 };
    std::size_t max_pending_requests { 64 };
    std::size_t max_batch_tokens { 128 };
    std::size_t kv_block_count { 64 };
    std::size_t kv_block_size { 16 };
};
enum class generation_errc {
    invalid_input,
    queue_full,
    execution_failure
};

struct generation_error {
    generation_errc kind;
    std::string message;
    std::string param;
    std::string code;
};

struct generation_request {
    std::vector<chat_message> messages;
    std::size_t max_completion_tokens;
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
    std::unique_ptr<text_decoder> decoder;
    std::size_t prompt_tokens {};
    std::size_t completion_tokens {};
    finish_reason reason { finish_reason::none };
    std::optional<generation_error> error;
    request_status status { request_status::submitted };
    std::atomic_bool cancelled {};
};

class serving_runtime {
public:
    static result<std::unique_ptr<serving_runtime>, inference_engine_errc> make(model_runner&,
                                                                                serving_config);
    ~serving_runtime();
    result<std::shared_ptr<request_state>, generation_error> submit(generation_request);
    void cancel(const std::shared_ptr<request_state>&) noexcept;

private:
    struct implementation;
    explicit serving_runtime(std::unique_ptr<implementation>);
    std::unique_ptr<implementation> implementation_;
};
} // namespace chibillm
