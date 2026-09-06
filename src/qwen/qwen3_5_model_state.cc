#include "qwen/qwen3_5_model_state.h"
#include "model_batch.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <new>
#include <unordered_set>

namespace chibillm {
namespace {
    result<tensor_descriptor, state_errc>
    f32_descriptor(std::vector<std::size_t> dimensions)
    {
        auto shape = tensor_shape::make(std::move(dimensions));
        if (!shape)
            return fail(state_errc::invalid_reservation);
        auto descriptor = tensor_descriptor::make(dtype::f32, std::move(*shape));
        if (!descriptor)
            return fail(state_errc::invalid_reservation);
        return std::move(*descriptor);
    }

    void
    zero(metal_tensor& tensor)
    {
        auto bytes = tensor.buffer().bytes();
        std::fill(bytes.begin(), bytes.end(), std::byte { 0 });
    }
} // namespace

result<std::unique_ptr<qwen3_5_model_state>, state_errc>
qwen3_5_model_state::make(const metal_context& context,
    const qwen3_5_config& config,
    std::size_t block_count,
    std::size_t block_size)
try {
    if (!config.layer_count
        || config.layer_types.size() != config.layer_count
        || !config.max_position_embeddings
        || !config.linear_key_head_count
        || !config.linear_key_head_dimension
        || !config.linear_value_head_count
        || !config.linear_value_head_dimension
        || !config.linear_conv_kernel_dimension
        || config.linear_value_head_count % config.linear_key_head_count != 0)
        return fail(state_errc::invalid_reservation);

    std::size_t full_count = 0, linear_count = 0;
    for (auto type : config.layer_types) {
        switch (type) {
        case qwen3_5_layer_type::linear_attention:
            ++linear_count;
            break;
        case qwen3_5_layer_type::full_attention:
            ++full_count;
            break;
        default:
            return fail(state_errc::invalid_reservation);
        }
    }

    // This state represents a hybrid model with both kinds of layers.
    if (!full_count || !linear_count)
        return fail(state_errc::invalid_reservation);

    constexpr auto max = std::numeric_limits<std::size_t>::max();
    if (config.linear_key_head_count > max / config.linear_key_head_dimension
        || config.linear_value_head_count > max / config.linear_value_head_dimension)
        return fail(state_errc::invalid_reservation);

    const auto kw = config.linear_key_width(), vw = config.linear_value_width();
    if (kw > (max - vw) / 2)
        return fail(state_errc::invalid_reservation);
    auto convolution = f32_descriptor({ 2 * kw + vw, config.linear_conv_kernel_dimension });
    auto recurrent = f32_descriptor({ config.linear_value_head_count, config.linear_key_head_dimension,
        config.linear_value_head_dimension });

    if (!convolution || !recurrent)
        return fail(state_errc::invalid_reservation);

    auto pages = block_manager::make(block_count, block_size);
    if (!pages)
        return fail(state_errc::invalid_reservation);

    auto cache = metal_kv_cache::make(
        context,
        { full_count, block_count, block_size, config.kv_head_count, config.head_dimension });

    if (!cache)
        return fail(cache.error() == kv_cache_errc::allocation_failed
                ? state_errc::backend_failure
                : state_errc::invalid_reservation);

    return std::unique_ptr<qwen3_5_model_state>(
        new qwen3_5_model_state(context, config, std::move(*pages), std::move(*cache),
            std::move(*convolution), std::move(*recurrent)));
} catch (const std::bad_alloc&) {
    return fail(state_errc::backend_failure);
}

qwen3_5_model_state::qwen3_5_model_state(const metal_context& context,
    qwen3_5_config config,
    block_manager pages,
    metal_kv_cache cache,
    tensor_descriptor convolution,
    tensor_descriptor recurrent)
    : context_(context)
    , config_(std::move(config))
    , pages_(std::move(pages))
    , cache_(std::move(cache))
    , convolution_descriptor_(std::move(convolution))
    , recurrent_descriptor_(std::move(recurrent))
    , linear_indices_(config_.layer_count)
    , cache_indices_(config_.layer_count)
{
    std::size_t full = 0;
    for (std::size_t layer = 0; layer < config_.layer_count; ++layer) {
        if (config_.layer_types[layer] == qwen3_5_layer_type::linear_attention)
            linear_indices_[layer] = linear_count_++;
        else
            cache_indices_[layer] = full++;
    }
}

result<void, state_errc>
qwen3_5_model_state::reserve(seq_id id, std::size_t tokens)
{
    if (batch_open_ || !tokens || tokens > config_.max_position_embeddings)
        return fail(state_errc::invalid_reservation);

    auto found = sequences_.find(id);
    const bool exists = found != sequences_.end();
    if (exists && tokens < found->second.reserved_tokens)
        return fail(state_errc::invalid_reservation);

    const auto required = 1 + (tokens - 1) / pages_.block_size();
    const auto existing = pages_.resources(id).blocks.size();
    if (required - existing > pages_.free_block_count())
        return fail(state_errc::capacity_exhausted);

    try {
        if (!exists) {
            sequence_state sequence;
            sequence.layers.reserve(linear_count_);
            for (std::size_t layer = 0; layer < linear_count_; ++layer) {
                auto convolution = metal_tensor::make(context_, convolution_descriptor_);
                auto recurrent = metal_tensor::make(context_, recurrent_descriptor_);
                if (!convolution || !recurrent)
                    return fail(state_errc::backend_failure);

                zero(*convolution);
                zero(*recurrent);
                sequence.layers.push_back({ std::move(*convolution), std::move(*recurrent) });
            }

            found = sequences_.emplace(id, std::move(sequence)).first;
        }

        auto reserved = pages_.reserve(id, tokens);
        if (!reserved) {
            if (!exists)
                sequences_.erase(id);

            return fail(reserved.error());
        }

        found->second.reserved_tokens = tokens;
        return { };
    } catch (const std::bad_alloc&) {
        if (!exists) {
            sequences_.erase(id);
            pages_.release(id);
        }

        return fail(state_errc::backend_failure);
    }
}

void qwen3_5_model_state::release(seq_id id) noexcept
{
    assert(!batch_open_);
    if (batch_open_)
        return;

    sequences_.erase(id);
    pages_.release(id);
}

std::size_t
qwen3_5_model_state::block_size() const noexcept
{
    return pages_.block_size();
}

sequence_resources
qwen3_5_model_state::resources(seq_id id) const noexcept
{
    return pages_.resources(id);
}

result<void, state_errc>
qwen3_5_model_state::begin_batch(const model_batch& batch)
try {
    if (batch_open_ || (batch.phase != batch_phase::prefill && batch.phase != batch_phase::decode))
        return fail(state_errc::invalid_reservation);
    if (!prepare_paged_batch(batch, config_.max_position_embeddings, cache_.block_count(),
            block_size()))
        return fail(state_errc::invalid_reservation);
    std::unordered_set<seq_id> seen;
    // Validate all sequences before copying state. Recurrent updates must append
    // exactly at the committed position; replaying a prefix would corrupt memory.
    for (const auto& item : batch.items) {
        const auto found = sequences_.find(item.id);
        if (found == sequences_.end()
            || !seen.insert(item.id).second
            || !std::ranges::equal(item.block_table, resources(item.id).blocks))
            return fail(state_errc::invalid_reservation);

        const auto& sequence = found->second;
        if (item.token_count > sequence.reserved_tokens - sequence.committed_tokens
            || (batch.phase == batch_phase::decode && item.token_count != 1))
            return fail(state_errc::invalid_reservation);

        for (std::size_t i = 0; i < item.token_count; ++i) {
            if (batch.positions[item.token_offset + i] != sequence.committed_tokens + i)
                return fail(state_errc::invalid_reservation);
        }
    }

    std::vector<sequence_snapshot> snapshots;
    snapshots.reserve(batch.items.size());
    for (const auto& item : batch.items) {
        const auto& sequence = sequences_.at(item.id);
        sequence_snapshot snapshot { item.id, sequence.committed_tokens + item.token_count, { } };
        snapshot.layers.reserve(linear_count_);

        for (const auto& layer : sequence.layers) {
            const auto conv = layer.convolution.buffer().bytes();
            const auto recurrent = layer.recurrent.buffer().bytes();
            snapshot.layers.push_back(
                { { conv.begin(), conv.end() }, { recurrent.begin(), recurrent.end() } });
        }
        snapshots.push_back(std::move(snapshot));
    }

    snapshots_ = std::move(snapshots);
    batch_open_ = true;
    return { };
} catch (const std::bad_alloc&) {
    return fail(state_errc::backend_failure);
}

void qwen3_5_model_state::commit_batch() noexcept
{
    if (!batch_open_)
        return;
    for (const auto& snapshot : snapshots_)
        sequences_.at(snapshot.id).committed_tokens = snapshot.end_position;
    snapshots_.clear();
    batch_open_ = false;
}

void qwen3_5_model_state::abort_batch() noexcept
{
    if (!batch_open_)
        return;

    for (const auto& snapshot : snapshots_) {
        auto& sequence = sequences_.at(snapshot.id);
        for (std::size_t layer = 0; layer < linear_count_; ++layer) {
            const auto& saved = snapshot.layers[layer];
            auto& target = sequence.layers[layer];
            std::memcpy(target.convolution.buffer().bytes().data(), saved.convolution.data(),
                saved.convolution.size());
            std::memcpy(target.recurrent.buffer().bytes().data(), saved.recurrent.data(),
                saved.recurrent.size());
        }
    }

    // KV writes only touched the uncommitted suffix. Its slots remain reserved
    // and are overwritten on retry, before they become visible to attention.
    snapshots_.clear();
    batch_open_ = false;
}

qwen3_5_linear_state*
qwen3_5_model_state::linear_state(seq_id id, std::size_t layer) noexcept
{
    if (layer >= linear_indices_.size() || !linear_indices_[layer])
        return nullptr;

    const auto found = sequences_.find(id);
    return found == sequences_.end() ? nullptr : &found->second.layers[*linear_indices_[layer]];
}

const qwen3_5_linear_state*
qwen3_5_model_state::linear_state(seq_id id, std::size_t layer) const noexcept
{
    if (layer >= linear_indices_.size() || !linear_indices_[layer])
        return nullptr;

    const auto found = sequences_.find(id);
    return found == sequences_.end() ? nullptr : &found->second.layers[*linear_indices_[layer]];
}

std::optional<std::size_t>
qwen3_5_model_state::cache_layer(std::size_t layer) const noexcept
{
    return layer < cache_indices_.size() ? cache_indices_[layer] : std::nullopt;
}

std::optional<std::size_t>
qwen3_5_model_state::committed_tokens(seq_id id) const noexcept
{
    const auto found = sequences_.find(id);
    return found == sequences_.end() ? std::nullopt : std::optional(found->second.committed_tokens);
}

std::size_t
qwen3_5_model_state::sequence_count() const noexcept
{
    return sequences_.size();
}

std::size_t
qwen3_5_model_state::linear_layer_count() const noexcept
{
    return linear_count_;
}

metal_kv_cache&
qwen3_5_model_state::cache() noexcept
{
    return cache_;
}

const metal_kv_cache&
qwen3_5_model_state::cache() const noexcept
{
    return cache_;
}
} // namespace chibillm
