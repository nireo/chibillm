#include "metal/metal_kv_cache.h"

#include <utility>
#include <vector>

#include "tensor/dtype.h"

namespace chibillm {

result<metal_kv_cache, kv_cache_errc>
metal_kv_cache::make(const metal_context& context, kv_cache_config config)
{
    if (config.layer_count == 0) {
        return fail(kv_cache_errc::invalid_layer_count);
    }
    if (config.block_count == 0) {
        return fail(kv_cache_errc::invalid_block_count);
    }
    if (config.block_size == 0) {
        return fail(kv_cache_errc::invalid_block_size);
    }
    if (config.kv_head_count == 0) {
        return fail(kv_cache_errc::invalid_kv_head_count);
    }
    if (config.head_dimension == 0) {
        return fail(kv_cache_errc::invalid_head_dimension);
    }

    std::vector<std::size_t> dimensions {
        config.layer_count,   config.block_count,    config.block_size,
        config.kv_head_count, config.head_dimension,
    };
    auto keys = metal_tensor::make(context, dtype::f32, dimensions);
    if (!keys) {
        return fail(keys.error() == metal_tensor_errc::invalid_descriptor
                        ? kv_cache_errc::layout_size_overflow
                        : kv_cache_errc::allocation_failed);
    }

    auto values = metal_tensor::make(context, dtype::f32, std::move(dimensions));
    if (!values) {
        return fail(values.error() == metal_tensor_errc::invalid_descriptor
                        ? kv_cache_errc::layout_size_overflow
                        : kv_cache_errc::allocation_failed);
    }

    return metal_kv_cache {
        config,
        std::move(*keys),
        std::move(*values),
    };
}

metal_kv_cache::metal_kv_cache(kv_cache_config config, metal_tensor keys, metal_tensor values)
    : config_(config)
    , keys_(std::move(keys))
    , values_(std::move(values))
{}

const kv_cache_config&
metal_kv_cache::config() const noexcept
{
    return config_;
}

std::size_t
metal_kv_cache::layer_count() const noexcept
{
    return config_.layer_count;
}

std::size_t
metal_kv_cache::block_count() const noexcept
{
    return config_.block_count;
}

std::size_t
metal_kv_cache::block_size() const noexcept
{
    return config_.block_size;
}

std::size_t
metal_kv_cache::kv_head_count() const noexcept
{
    return config_.kv_head_count;
}

std::size_t
metal_kv_cache::head_dimension() const noexcept
{
    return config_.head_dimension;
}

std::size_t
metal_kv_cache::elements_per_token() const noexcept
{
    return config_.kv_head_count * config_.head_dimension;
}

std::size_t
metal_kv_cache::elements_per_block() const noexcept
{
    return config_.block_size * elements_per_token();
}

std::size_t
metal_kv_cache::elements_per_layer() const noexcept
{
    return config_.block_count * elements_per_block();
}

std::size_t
metal_kv_cache::element_count() const noexcept
{
    return keys_.descriptor().element_count();
}

result<std::size_t, kv_cache_errc>
metal_kv_cache::element_offset(std::size_t layer,
                               std::size_t block,
                               std::size_t token_offset,
                               std::size_t kv_head,
                               std::size_t head_feature) const noexcept
{
    if (layer >= config_.layer_count) {
        return fail(kv_cache_errc::layer_out_of_range);
    }
    if (block >= config_.block_count) {
        return fail(kv_cache_errc::block_out_of_range);
    }
    if (token_offset >= config_.block_size) {
        return fail(kv_cache_errc::token_offset_out_of_range);
    }
    if (kv_head >= config_.kv_head_count) {
        return fail(kv_cache_errc::kv_head_out_of_range);
    }
    if (head_feature >= config_.head_dimension) {
        return fail(kv_cache_errc::head_feature_out_of_range);
    }

    auto offset = layer;
    offset = offset * config_.block_count + block;
    offset = offset * config_.block_size + token_offset;
    offset = offset * config_.kv_head_count + kv_head;
    offset = offset * config_.head_dimension + head_feature;
    return offset;
}

metal_tensor&
metal_kv_cache::keys() noexcept
{
    return keys_;
}

const metal_tensor&
metal_kv_cache::keys() const noexcept
{
    return keys_;
}

metal_tensor&
metal_kv_cache::values() noexcept
{
    return values_;
}

const metal_tensor&
metal_kv_cache::values() const noexcept
{
    return values_;
}

} // namespace chibillm
