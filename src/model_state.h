#pragma once
#include "seq.h"
#include <span>

namespace chibillm {
struct model_batch;
enum class state_errc {
    capacity_exhausted,
    invalid_reservation,
    backend_failure
};

struct sequence_resources {
    std::span<const block_id> blocks;
};

class model_state {
public:
    virtual ~model_state() = default;
    virtual result<void, state_errc> reserve(seq_id id, std::size_t token_count) = 0;
    virtual void release(seq_id id) noexcept = 0;

    virtual std::size_t
    block_size() const noexcept
    {
        return 0;
    }

    virtual sequence_resources
    resources(seq_id) const noexcept
    {
        return {};
    }

    // Failed execution must restore pre-batch state before the reservation can be retried.
    virtual result<void, state_errc>
    begin_batch(const model_batch&)
    {
        return {};
    }

    virtual void
    commit_batch() noexcept
    {}

    virtual void
    abort_batch() noexcept
    {}
};
} // namespace chibillm
