#include "qwen/qwen_embedding.h"

#include "tensor/dtype.h"
#include "tensor/tensor_ops.h"
#include <cstring>

namespace chibillm {
namespace {

qwen_embedding_errc
tensor_error(metal_tensor_errc error)
{
    return error == metal_tensor_errc::invalid_descriptor
        ? qwen_embedding_errc::tensor_creation_failed
        : qwen_embedding_errc::metal_allocation_failed;
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
    auto token_ids = metal_tensor::make(context, dtype::i32, { tokens.size() });
    if (!token_ids) {
        return fail(tensor_error(token_ids.error()));
    }
    auto output = metal_tensor::make(context, dtype::f32, { tokens.size(), hidden_size });
    if (!output) {
        return fail(tensor_error(output.error()));
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
