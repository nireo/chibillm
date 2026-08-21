#pragma once

#include <cstdint>
#include <vector>

#include "model_batch.h"
#include "result.h"

namespace chibillm {

enum class model_runner_errc : std::uint8_t {
    empty_batch,
    inconsistent_batch,
    backend_failure,
};

// executes one model batch and returns one sample slot per sequence.
class model_runner {
public:
    virtual ~model_runner() = default;

    [[nodiscard]] virtual result<std::vector<token_id>, model_runner_errc>
    execute(const model_batch& batch) = 0;
};

// returns a fixed token while the real model backend is unavailable.
class fake_model_runner final : public model_runner {
public:
    explicit fake_model_runner(token_id output_token) noexcept;

    [[nodiscard]] result<std::vector<token_id>, model_runner_errc>
    execute(const model_batch& batch) override;

private:
    token_id output_token_;
};

} // namespace chibillm
