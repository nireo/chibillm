#pragma once

#include "error.h"

#include <cstddef>
#include <cstdint>

#include "metal/metal_context.h"
#include "metal/metal_tensor.h"
#include "result.h"

namespace chibillm {

// describes one shared key/value cache allocation.
struct kv_cache_config {
    std::size_t layer_count;
    std::size_t block_count;
    std::size_t block_size;
    std::size_t kv_head_count;
    std::size_t head_dimension;
};

enum class kv_cache_errc : std::uint8_t {
    invalid_layer_count,
    invalid_block_count,
    invalid_block_size,
    invalid_kv_head_count,
    invalid_head_dimension,
    layout_size_overflow,
    allocation_failed,
    layer_out_of_range,
    block_out_of_range,
    token_offset_out_of_range,
    kv_head_out_of_range,
    head_feature_out_of_range,
};

[[nodiscard]] inline std::string_view
error_name(kv_cache_errc code) noexcept
{
    static constexpr std::array names {
        "kv_cache.invalid_layer_count",    "kv_cache.invalid_block_count",
        "kv_cache.invalid_block_size",     "kv_cache.invalid_kv_head_count",
        "kv_cache.invalid_head_dimension", "kv_cache.layout_size_overflow",
        "kv_cache.allocation_failed",      "kv_cache.layer_out_of_range",
        "kv_cache.block_out_of_range",     "kv_cache.token_offset_out_of_range",
        "kv_cache.kv_head_out_of_range",   "kv_cache.head_feature_out_of_range",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "kv_cache.unknown_error";
}

using kv_cache_error = error<kv_cache_errc>;

// owns f32 key and value tensors with the same paged layout.
class metal_kv_cache {
public:
    [[nodiscard]] static result<metal_kv_cache, kv_cache_error> make(const metal_context& context,
                                                                     kv_cache_config config);

    metal_kv_cache(const metal_kv_cache&) = delete;
    metal_kv_cache& operator=(const metal_kv_cache&) = delete;
    metal_kv_cache(metal_kv_cache&&) noexcept = default;
    metal_kv_cache& operator=(metal_kv_cache&&) noexcept = default;

    [[nodiscard]] const kv_cache_config& config() const noexcept;
    [[nodiscard]] std::size_t layer_count() const noexcept;
    [[nodiscard]] std::size_t block_count() const noexcept;
    [[nodiscard]] std::size_t block_size() const noexcept;
    [[nodiscard]] std::size_t kv_head_count() const noexcept;
    [[nodiscard]] std::size_t head_dimension() const noexcept;
    [[nodiscard]] std::size_t elements_per_token() const noexcept;
    [[nodiscard]] std::size_t elements_per_block() const noexcept;
    [[nodiscard]] std::size_t elements_per_layer() const noexcept;
    [[nodiscard]] std::size_t element_count() const noexcept;

    // returns an element index into either flat cache tensor.
    [[nodiscard]] result<std::size_t, kv_cache_error>
    element_offset(std::size_t layer,
                   std::size_t block,
                   std::size_t token_offset,
                   std::size_t kv_head,
                   std::size_t head_feature) const noexcept;

    [[nodiscard]] metal_tensor& keys() noexcept;
    [[nodiscard]] const metal_tensor& keys() const noexcept;
    [[nodiscard]] metal_tensor& values() noexcept;
    [[nodiscard]] const metal_tensor& values() const noexcept;

private:
    metal_kv_cache(kv_cache_config config, metal_tensor keys, metal_tensor values);

    kv_cache_config config_;
    metal_tensor keys_;
    metal_tensor values_;
};

} // namespace chibillm
