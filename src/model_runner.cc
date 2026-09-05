#include "model_runner.h"

#include "block_manager.h"

namespace chibillm {

result<std::unique_ptr<model_state>, model_runner_errc>
model_runner::make_state(scheduler_config config) const
{
    auto state = block_manager::make(config.kv_block_count, config.kv_block_size);
    if (!state)
        return fail(model_runner_errc::backend_failure);
    return std::make_unique<block_manager>(std::move(*state));
}

} // namespace chibillm
