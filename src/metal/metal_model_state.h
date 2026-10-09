#pragma once
#include "block_manager.h"
#include "metal/metal_kv_cache.h"

namespace chibillm {
class metal_model_state final : public model_state {
public:
    static result<std::unique_ptr<metal_model_state>, state_error>
    make(const metal_context& context, kv_cache_config config)
    {
        auto pages = block_manager::make(config.block_count, config.block_size);
        if (!pages)
            return fail(state_errc::invalid_reservation, pages.error(), "block allocator");
        auto cache = metal_kv_cache::make(context, config);
        if (!cache)
            return fail(state_errc::backend_failure, cache.error(), "KV cache");
        return std::unique_ptr<metal_model_state>(
            new metal_model_state(std::move(*pages), std::move(*cache)));
    }

    result<void, state_error>
    reserve(seq_id id, std::size_t tokens) override
    {
        return pages_.reserve(id, tokens);
    }

    void
    release(seq_id id) noexcept override
    {
        pages_.release(id);
    }

    std::size_t
    block_size() const noexcept override
    {
        return pages_.block_size();
    }

    sequence_resources
    resources(seq_id id) const noexcept override
    {
        return pages_.resources(id);
    }

    std::size_t
    checkpoint_bytes(std::size_t tokens) const noexcept override
    {
        return cache_.checkpoint_bytes(tokens);
    }

    result<std::unique_ptr<model_checkpoint>, state_error>
    checkpoint(seq_id id, std::size_t tokens) const override
    {
        auto saved = cache_.checkpoint(resources(id).blocks, tokens);
        if (!saved)
            return fail(saved.error());
        return std::move(*saved);
    }

    result<void, state_error>
    restore(seq_id id, const model_checkpoint& checkpoint) override
    {
        const auto* saved = dynamic_cast<const kv_cache_checkpoint*>(&checkpoint);
        if (!saved)
            return fail(state_errc::invalid_reservation);
        CL_TRY(reserve(id, saved->token_count));
        return cache_.restore(resources(id).blocks, *saved);
    }

    metal_kv_cache&
    cache()
    {
        return cache_;
    }

private:
    metal_model_state(block_manager pages, metal_kv_cache cache)
        : pages_(std::move(pages))
        , cache_(std::move(cache))
    {}

    block_manager pages_;
    metal_kv_cache cache_;
};
} // namespace chibillm
