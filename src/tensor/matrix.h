#pragma once

#include <variant>

#include "metal/metal_tensor.h"
#include "tensor/weight_format.h"

namespace chibillm {

enum class matrix_errc : std::uint8_t {
    invalid_layout,
    nonfinite_weight,
    allocation_failed,
};

[[nodiscard]] inline std::string_view
error_name(matrix_errc code) noexcept
{
    static constexpr std::array names {
        "matrix.invalid_layout",
        "matrix.nonfinite_weight",
        "matrix.allocation_failed",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "matrix.unknown_error";
}

using matrix_error = error<matrix_errc>;

// Row-major affine Q4: eight consecutive values per U32, one BF16 scale and
// offset per 64 input features. Logical dimensions never describe packed data.
class quantized_matrix {
public:
    static constexpr std::size_t group_size = 64;
    static constexpr std::size_t values_per_word = 8;

    [[nodiscard]] static result<quantized_matrix, matrix_error>
    quantize(const metal_context& context, const metal_tensor& source);

    [[nodiscard]] const tensor_shape&
    shape() const noexcept
    {
        return shape_;
    }

    [[nodiscard]] const metal_tensor&
    packed() const noexcept
    {
        return packed_;
    }

    [[nodiscard]] const metal_tensor&
    scales() const noexcept
    {
        return scales_;
    }

    [[nodiscard]] const metal_tensor&
    offsets() const noexcept
    {
        return offsets_;
    }

    [[nodiscard]] std::size_t
    size_bytes() const noexcept
    {
        return packed_.buffer().size_bytes()
            + scales_.buffer().size_bytes()
            + offsets_.buffer().size_bytes();
    }

private:
    quantized_matrix(tensor_shape shape,
                     metal_tensor packed,
                     metal_tensor scales,
                     metal_tensor offsets)
        : shape_(std::move(shape))
        , packed_(std::move(packed))
        , scales_(std::move(scales))
        , offsets_(std::move(offsets))
    {}

    tensor_shape shape_;
    metal_tensor packed_;
    metal_tensor scales_;
    metal_tensor offsets_;
};

// Non-owning operation argument. Existing dense tensors and owned model matrices
// use the same projection API without copying storage or erasing logical shape.
class matrix_view {
public:
    matrix_view(const metal_tensor& tensor)
        : storage_(&tensor)
    {}

    matrix_view(const quantized_matrix& matrix)
        : storage_(&matrix)
    {}

    [[nodiscard]] const metal_tensor*
    dense() const noexcept
    {
        const auto* value = std::get_if<const metal_tensor*>(&storage_);
        return value ? *value : nullptr;
    }

    [[nodiscard]] const quantized_matrix*
    quantized() const noexcept
    {
        const auto* value = std::get_if<const quantized_matrix*>(&storage_);
        return value ? *value : nullptr;
    }

    [[nodiscard]] const tensor_shape&
    shape() const noexcept
    {
        if (const auto* value = dense())
            return value->descriptor().shape();
        return quantized()->shape();
    }

    [[nodiscard]] bool
    supported() const noexcept
    {
        return shape().rank() == 2 && (!dense() || dense()->descriptor().type() == dtype::bf16);
    }

    [[nodiscard]] std::size_t
    size_bytes() const noexcept
    {
        return dense() ? dense()->buffer().size_bytes() : quantized()->size_bytes();
    }

private:
    std::variant<const metal_tensor*, const quantized_matrix*> storage_;
};

class matrix_weight {
public:
    matrix_weight(metal_tensor tensor)
        : storage_(std::move(tensor))
    {}

    matrix_weight(quantized_matrix matrix)
        : storage_(std::move(matrix))
    {}

    [[nodiscard]]
    operator matrix_view() const noexcept
    {
        return std::visit([](const auto& value) { return matrix_view(value); }, storage_);
    }

    [[nodiscard]] const tensor_shape&
    shape() const noexcept
    {
        return matrix_view(*this).shape();
    }

    [[nodiscard]] std::size_t
    size_bytes() const noexcept
    {
        return matrix_view(*this).size_bytes();
    }

    [[nodiscard]] metal_tensor*
    dense() noexcept
    {
        return std::get_if<metal_tensor>(&storage_);
    }

    [[nodiscard]] const metal_tensor*
    dense() const noexcept
    {
        return std::get_if<metal_tensor>(&storage_);
    }

private:
    std::variant<metal_tensor, quantized_matrix> storage_;
};

} // namespace chibillm
