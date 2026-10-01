#pragma once
#include "model_runner.h"

namespace chibillm {
result<std::string, model_runner_error> format_qwen_chat(std::span<const chat_message> messages,
                                                         bool thinking = false);
}
