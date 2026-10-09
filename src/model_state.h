#pragma once

#include "error.h"
#include "seq.h"
#include <memory>
#include <span>

namespace chibillm {
struct model_batch;
enum class state_errc {
    capacity_exhausted,
    invalid_reservation,
    backend_failure,
    allocation_failed,
};

[[nodiscard]] inline std::string_view
error_name(state_errc code) noexcept
{
    static constexpr std::array names {
        "state.capacity_exhausted",
        "state.invalid_reservation",
        "state.backend_failure",
        "state.allocation_failed",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "state.unknown_error";
}

using state_error = error<state_errc>;

struct sequence_resources {
    std::span<const block_id> blocks;
};

// An immutable, backend-owned copy of the state after an exact token prefix.
struct model_checkpoint {
    explicit model_checkpoint(std::size_t token_count) noexcept
        : token_count(token_count)
    {}

    virtual ~model_checkpoint() = default;
    const std::size_t token_count;
    virtual std::size_t size_bytes() const noexcept = 0;
};

class model_state {
public:
    virtual ~model_state() = default;
    // Ensure capacity for at least token_count tokens; smaller retries keep existing capacity.
    virtual result<void, state_error> reserve(seq_id id, std::size_t token_count) = 0;
    virtual void release(seq_id id) noexcept = 0;

    // Zero means checkpointing is unsupported or this prefix is out of range.
    virtual std::size_t
    checkpoint_bytes(std::size_t) const noexcept
    {
        return 0;
    }

    virtual result<std::unique_ptr<model_checkpoint>, state_error>
    checkpoint(seq_id, std::size_t) const
    {
        return std::unique_ptr<model_checkpoint> {};
    }

    // Restore into a fresh reservation, between completed GPU batches.
    virtual result<void, state_error>
    restore(seq_id, const model_checkpoint&)
    {
        return fail(state_errc::invalid_reservation);
    }

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
    virtual result<void, state_error>
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
