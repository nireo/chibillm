#include "qwen/qwen_embedding.h"

#include <cstring>
#include <utility>
#include <vector>

#include "tensor/dtype.h"
#include "tensor/tensor_descriptor.h"
#include "tensor/tensor_ops.h"
#include "tensor/tensor_shape.h"

namespace chibillm {
namespace {

result<metal_tensor, qwen_embedding_errc>
make_tensor(const metal_context& context, dtype type, std::vector<std::size_t> dimensions)
{
    auto shape = tensor_shape::make(std::move(dimensions));
    if (!shape) {
        return fail(qwen_embedding_errc::tensor_creation_failed);
    }
    auto descriptor = tensor_descriptor::make(type, std::move(*shape));
    if (!descriptor) {
        return fail(qwen_embedding_errc::tensor_creation_failed);
    }
    auto tensor = metal_tensor::make(context, std::move(*descriptor));
    if (!tensor) {
        return fail(qwen_embedding_errc::metal_allocation_failed);
    }
    return std::move(*tensor);
}

} // namespace

result<metal_tensor, qwen_embedding_errc>
embed_qwen_tokens(const metal_context& context,
                  const qwen_weights& weights,
                  std::span<const token_id> tokens)
{
    if (tokens.empty()) {
        return fail(qwen_embedding_errc::empty_tokens);
    }

    const auto hidden_size = weights.token_embedding.descriptor().shape().dimensions()[1];
    auto token_ids = make_tensor(context, dtype::i32, { tokens.size() });
    if (!token_ids) {
        return fail(token_ids.error());
    }
    auto output = make_tensor(context, dtype::f32, { tokens.size(), hidden_size });
    if (!output) {
        return fail(output.error());
    }

    std::memcpy(token_ids->buffer().bytes().data(), tokens.data(), tokens.size_bytes());
    auto embedded = embedding_lookup(context, *token_ids, weights.token_embedding, *output);
    if (!embedded) {
        return fail(embedded.error() == tensor_op_errc::token_out_of_range
                        ? qwen_embedding_errc::invalid_token
                        : qwen_embedding_errc::backend_failure);
    }
    return std::move(*output);
}

} // namespace chibillm
