#pragma once
#include "model_runner.h"

namespace chibillm {
result<std::string, model_runner_errc> format_qwen_chat(std::span<const chat_message> messages,
                                                        bool thinking = false);
}
