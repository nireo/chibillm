#include "qwen/qwen_chat.h"

namespace chibillm {
result<std::string, model_runner_errc>
format_qwen_chat(std::span<const chat_message> messages, bool thinking)
{
    if (messages.empty()) {
        return fail(model_runner_errc::invalid_chat);
    }

    std::string prompt;
    for (const auto& message : messages) {
        if (message.role != "developer"
            && message.role != "system"
            && message.role != "user"
            && message.role != "assistant") {
            return fail(model_runner_errc::invalid_chat);
        }
        const auto role = message.role == "developer" ? "system" : message.role;
        prompt += "<|im_start|>" + role + "\n" + message.content + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";
    if (!thinking)
        prompt += "<think>\n\n</think>\n\n";

    return prompt;
}
} // namespace chibillm
