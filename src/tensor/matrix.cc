#include "tensor/matrix.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace chibillm {

result<quantized_matrix, matrix_error>
quantized_matrix::from_packed(tensor_shape shape,
                              metal_tensor packed,
                              metal_tensor scales,
                              metal_tensor offsets)
{
    if (shape.rank() != 2 || shape.dimensions()[1] % group_size != 0)
        return fail(matrix_errc::invalid_layout);
    const auto rows = shape.dimensions()[0], columns = shape.dimensions()[1];
    const auto matches = [](const metal_tensor& tensor, dtype type, std::size_t n, std::size_t k) {
        const auto& descriptor = tensor.descriptor();
        return descriptor.type() == type
            && descriptor.shape().rank() == 2
            && descriptor.shape().dimensions()[0] == n
            && descriptor.shape().dimensions()[1] == k;
    };
    if (!matches(packed, dtype::u32, rows, columns / values_per_word)
        || !matches(scales, dtype::bf16, rows, columns / group_size)
        || !matches(offsets, dtype::bf16, rows, columns / group_size))
        return fail(matrix_errc::invalid_layout);
    for (const auto* tensor : { &scales, &offsets }) {
        const auto bytes = tensor->buffer().bytes();
        for (std::size_t i = 0; i < bytes.size(); i += sizeof(std::uint16_t)) {
            std::uint16_t bits;
            std::memcpy(&bits, bytes.data() + i, sizeof(bits));
            if (!std::isfinite(bf16::from_bits(bits).to_float()))
                return fail(matrix_errc::nonfinite_weight);
        }
    }
    return quantized_matrix(std::move(shape), std::move(packed), std::move(scales),
                            std::move(offsets));
}

result<quantized_matrix, matrix_error>
quantized_matrix::quantize(const metal_context& context, const metal_tensor& source)
{
    const auto& shape = source.descriptor().shape();
    if (shape.rank() != 2
        || source.descriptor().type() != dtype::bf16
        || shape.dimensions()[1] % group_size != 0)
        return fail(matrix_errc::invalid_layout);

    const auto rows = shape.dimensions()[0];
    const auto columns = shape.dimensions()[1];
    auto packed = metal_tensor::make(context, dtype::u32, { rows, columns / values_per_word });
    auto scales = metal_tensor::make(context, dtype::bf16, { rows, columns / group_size });
    auto offsets = metal_tensor::make(context, dtype::bf16, { rows, columns / group_size });
    if (!packed)
        return fail(matrix_errc::allocation_failed, packed.error(), "packed weights");
    if (!scales)
        return fail(matrix_errc::allocation_failed, scales.error(), "scales");
    if (!offsets)
        return fail(matrix_errc::allocation_failed, offsets.error(), "offsets");

    const auto input = source.buffer().bytes();
    auto output = packed->buffer().bytes();
    auto scale_bytes = scales->buffer().bytes();
    auto offset_bytes = offsets->buffer().bytes();
    for (std::size_t group = 0; group < shape.element_count() / group_size; ++group) {
        std::array<float, group_size> values;
        for (std::size_t i = 0; i < group_size; ++i) {
            std::uint16_t bits;
            std::memcpy(&bits, input.data() + (group * group_size + i) * sizeof(bits),
                        sizeof(bits));
            values[i] = bf16::from_bits(bits).to_float();
            if (!std::isfinite(values[i]))
                return fail(matrix_errc::nonfinite_weight);
        }
        const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
        const auto offset = bf16::from_float(*minimum);
        // Calculate in double so a finite BF16 range cannot overflow subtraction.
        const auto scale =
            bf16::from_float(static_cast<float>((double(*maximum) - *minimum) / 15.0));
        const auto scale_value = scale.to_float();
        const auto offset_value = offset.to_float();
        const auto scale_bits = scale.bits();
        const auto offset_bits = offset.bits();
        std::memcpy(scale_bytes.data() + group * sizeof(scale_bits), &scale_bits,
                    sizeof(scale_bits));
        std::memcpy(offset_bytes.data() + group * sizeof(offset_bits), &offset_bits,
                    sizeof(offset_bits));

        for (std::size_t word = 0; word < group_size / values_per_word; ++word) {
            std::uint32_t bits = 0;
            for (std::size_t i = 0; i < values_per_word; ++i) {
                const auto q = scale_value == 0.0F
                    ? 0.0
                    : std::clamp(
                          std::round((double(values[word * values_per_word + i]) - offset_value)
                                     / scale_value),
                          0.0, 15.0);
                bits |= static_cast<std::uint32_t>(q) << (i * 4);
            }
            std::memcpy(output.data()
                            + (group * group_size / values_per_word + word) * sizeof(bits),
                        &bits, sizeof(bits));
        }
    }
    return quantized_matrix(shape, std::move(*packed), std::move(*scales), std::move(*offsets));
}

} // namespace chibillm
