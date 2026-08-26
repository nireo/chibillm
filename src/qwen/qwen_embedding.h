#pragma once

#include <cstdint>
#include <span>

#include "metal/metal_context.h"
#include "metal/metal_tensor.h"
#include "qwen/qwen_weights.h"
#include "result.h"
#include "seq.h"

namespace chibillm {

enum class qwen_embedding_errc : std::uint8_t {
    empty_tokens,
    tensor_creation_failed,
    metal_allocation_failed,
    invalid_token,
    backend_failure,
};

[[nodiscard]] result<metal_tensor, qwen_embedding_errc> embed_qwen_tokens(
    const metal_context& context, const qwen_weights& weights, std::span<const token_id> tokens);

} // namespace chibillm
