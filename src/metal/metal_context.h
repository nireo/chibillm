#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

#include "metal/metal_buffer.h"
#include "metal/metal_error.h"
#include "result.h"

namespace chibillm {

class metal_tensor;
enum class tensor_op_errc : std::uint8_t;

// owns the metal device, queue, shader library, and compute pipelines.
class metal_context {
public:
    [[nodiscard]] static result<metal_context, metal_error> make(std::string_view shader_source);

    metal_context(const metal_context&) = delete;
    metal_context& operator=(const metal_context&) = delete;
    metal_context(metal_context&&) noexcept;
    metal_context& operator=(metal_context&&) noexcept;
    ~metal_context();

    [[nodiscard]] std::string_view device_name() const noexcept;

    [[nodiscard]] result<metal_buffer, metal_error>
    make_shared_buffer(std::size_t size_bytes) const;

private:
    friend result<void, tensor_op_errc> matmul(const metal_context& context,
                                               const metal_tensor& lhs,
                                               const metal_tensor& rhs,
                                               metal_tensor& output);

    friend result<void, tensor_op_errc> linear(const metal_context& context,
                                               const metal_tensor& input,
                                               const metal_tensor& weight,
                                               metal_tensor& output);

    friend result<void, tensor_op_errc> embedding_lookup(const metal_context& context,
                                                         const metal_tensor& token_ids,
                                                         const metal_tensor& weight,
                                                         metal_tensor& output);

    struct implementation;

    explicit metal_context(std::unique_ptr<implementation> implementation) noexcept;

    [[nodiscard]] result<void, metal_error> dispatch_matmul(const metal_buffer& lhs,
                                                            const metal_buffer& rhs,
                                                            metal_buffer& output,
                                                            std::size_t rows,
                                                            std::size_t inner_dimension,
                                                            std::size_t columns) const;

    [[nodiscard]] result<void, metal_error> dispatch_linear_bf16(const metal_buffer& input,
                                                                 const metal_buffer& weight,
                                                                 metal_buffer& output,
                                                                 std::size_t rows,
                                                                 std::size_t input_features,
                                                                 std::size_t output_features) const;

    [[nodiscard]] result<void, metal_error> dispatch_embedding_bf16(const metal_buffer& token_ids,
                                                                    const metal_buffer& weight,
                                                                    metal_buffer& output,
                                                                    std::size_t token_count,
                                                                    std::size_t hidden_size) const;

    std::unique_ptr<implementation> implementation_;
};

} // namespace chibillm
