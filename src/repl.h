#pragma once

#include <cstddef>

namespace chibillm {

class model_runner;
struct scheduler_config;

int run_repl(model_runner& runner, scheduler_config config, std::size_t max_new_tokens);

} // namespace chibillm
