#include "metal/metal_kv_cache.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "tensor/types.h"

namespace chibillm {
namespace {
void
copy_prefix(std::span<const std::byte> source,
            std::span<std::byte> target,
            std::span<const block_id> blocks,
            const metal_kv_cache& cache,
            std::size_t tokens,
            bool restore)
{
    const auto token_bytes = cache.elements_per_token() * sizeof(float);
    const auto block_bytes = cache.elements_per_block() * sizeof(float);
    const auto layer_bytes = cache.elements_per_layer() * sizeof(float);
    for (std::size_t layer = 0; layer < cache.layer_count(); ++layer) {
        for (std::size_t token = 0; token < tokens; token += cache.block_size()) {
            const auto page = blocks[token / cache.block_size()];
            const auto paged_offset = layer * layer_bytes + page * block_bytes;
            const auto packed_offset = (layer * tokens + token) * token_bytes;
            const auto count = std::min(cache.block_size(), tokens - token) * token_bytes;
            std::memcpy(target.data() + (restore ? paged_offset : packed_offset),
                        source.data() + (restore ? packed_offset : paged_offset), count);
        }
    }
}
} // namespace

result<metal_kv_cache, kv_cache_error>
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
                        : kv_cache_errc::allocation_failed,
                    keys.error(), "KV keys");
    }

    auto values = metal_tensor::make(context, dtype::f32, std::move(dimensions));
    if (!values) {
        return fail(values.error() == metal_tensor_errc::invalid_descriptor
                        ? kv_cache_errc::layout_size_overflow
                        : kv_cache_errc::allocation_failed,
                    values.error(), "KV values");
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

result<std::size_t, kv_cache_error>
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

std::size_t
metal_kv_cache::checkpoint_bytes(std::size_t tokens) const noexcept
{
    if (!tokens || tokens > block_count() * block_size())
        return 0;
    const auto bytes = tokens * elements_per_token() * layer_count() * sizeof(float);
    return bytes <= std::numeric_limits<std::size_t>::max() / 2 ? 2 * bytes : 0;
}

bool
metal_kv_cache::valid_prefix(std::span<const block_id> blocks, std::size_t tokens) const noexcept
{
    if (!checkpoint_bytes(tokens))
        return false;
    const auto required = 1 + (tokens - 1) / block_size();
    return blocks.size() >= required
        && std::ranges::all_of(blocks.first(required),
                               [&](auto block) { return block < block_count(); });
}

result<std::unique_ptr<kv_cache_checkpoint>, state_error>
metal_kv_cache::checkpoint(std::span<const block_id> blocks, std::size_t tokens) const
try {
    if (!valid_prefix(blocks, tokens))
        return fail(state_errc::invalid_reservation);
    auto saved = std::make_unique<kv_cache_checkpoint>(config_, tokens);
    saved->keys.resize(checkpoint_bytes(tokens) / 2);
    saved->values.resize(saved->keys.size());
    copy_prefix(keys_.buffer().bytes(), saved->keys, blocks, *this, tokens, false);
    copy_prefix(values_.buffer().bytes(), saved->values, blocks, *this, tokens, false);
    return saved;
} catch (const std::bad_alloc&) {
    return fail(state_errc::allocation_failed);
}

result<void, state_error>
metal_kv_cache::restore(std::span<const block_id> blocks, const kv_cache_checkpoint& saved)
{
    if (!valid_prefix(blocks, saved.token_count)
        || saved.config.layer_count != layer_count()
        || saved.config.kv_head_count != kv_head_count()
        || saved.config.head_dimension != head_dimension()
        || saved.keys.size() != checkpoint_bytes(saved.token_count) / 2
        || saved.values.size() != saved.keys.size())
        return fail(state_errc::invalid_reservation);
    copy_prefix(saved.keys, keys_.buffer().bytes(), blocks, *this, saved.token_count, true);
    copy_prefix(saved.values, values_.buffer().bytes(), blocks, *this, saved.token_count, true);
    return {};
}

} // namespace chibillm
