#pragma once

#include <bit>
#include <cstdint>

namespace chibillm {

class bf16 {
public:
    [[nodiscard]] static bf16 from_bits(std::uint16_t bits) noexcept;
    [[nodiscard]] static bf16 from_float(float value) noexcept;

    [[nodiscard]] std::uint16_t bits() const noexcept;
    [[nodiscard]] float to_float() const noexcept;

private:
    explicit bf16(std::uint16_t bits) noexcept;

    std::uint16_t bits_;
};

inline bf16::bf16(std::uint16_t bits) noexcept
    : bits_(bits)
{}

inline bf16
bf16::from_bits(std::uint16_t bits) noexcept
{
    return bf16 { bits };
}

inline bf16
bf16::from_float(float value) noexcept
{
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);

    // NaN: preserve the sign/payload bits that fit and ensure the bf16 mantissa remains non-zero.
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return bf16(static_cast<std::uint16_t>((bits >> 16) | 0x0040u));
    }

    // round to nearest, ties to even.
    const std::uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    bits += rounding_bias;

    return bf16(static_cast<std::uint16_t>(bits >> 16));
}

inline std::uint16_t
bf16::bits() const noexcept
{
    return bits_;
}

inline float
bf16::to_float() const noexcept
{
    const auto float_bits = static_cast<std::uint32_t>(bits_) << 16U;
    return std::bit_cast<float>(float_bits);
}

} // namespace chibillm
