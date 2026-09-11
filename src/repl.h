#pragma once

#include <cstddef>
#include <iosfwd>

namespace chibillm {

class model_runner;
struct scheduler_config;

int run_repl(model_runner& runner,
             scheduler_config config,
             std::size_t max_new_tokens,
             bool stream = true,
             bool progress = true,
             std::ostream* metrics_output = nullptr);

} // namespace chibillm
