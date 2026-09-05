#include "tensor/embedding.h"

#include "tensor/dtype.h"
#include "tensor/tensor_ops.h"
#include <cstring>

namespace chibillm {

result<metal_tensor, tensor_op_errc>
embed_tokens(const metal_context& context,
             const metal_tensor& weight,
             std::span<const token_id> tokens)
{
    if (tokens.empty()) {
        return fail(tensor_op_errc::empty_tokens);
    }

    const auto hidden_size = weight.descriptor().shape().dimensions()[1];
    auto token_ids = allocate_tensor(context, dtype::i32, { tokens.size() });
    if (!token_ids) {
        return fail(token_ids.error());
    }
    auto output = allocate_tensor(context, dtype::f32, { tokens.size(), hidden_size });
    if (!output) {
        return fail(output.error());
    }

    std::memcpy(token_ids->buffer().bytes().data(), tokens.data(), tokens.size_bytes());
    auto embedded = embedding_lookup(context, *token_ids, weight, *output);
    if (!embedded) {
        return fail(embedded.error() == tensor_op_errc::token_out_of_range
                        ? tensor_op_errc::token_out_of_range
                        : tensor_op_errc::backend_failure);
    }
    return std::move(*output);
}

} // namespace chibillm
