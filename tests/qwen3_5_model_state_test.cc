#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "metal_test_support.h"
#include "model_batch.h"
#include "qwen/qwen3_5_model_state.h"
#include "tensor/deltanet.h"

#include <algorithm>
#include <limits>

using namespace chibillm;
using namespace metal_test;

namespace {
qwen3_5_config
small_config()
{
    qwen3_5_config c {};
    c.layer_count = 4;
    c.layer_types = { qwen3_5_layer_type::linear_attention, qwen3_5_layer_type::full_attention,
                      qwen3_5_layer_type::linear_attention, qwen3_5_layer_type::full_attention };
    c.max_position_embeddings = 32;
    c.kv_head_count = 1;
    c.head_dimension = 2;
    c.linear_key_head_count = 1;
    c.linear_value_head_count = 1;
    c.linear_key_head_dimension = 2;
    c.linear_value_head_dimension = 2;
    c.linear_conv_kernel_dimension = 4;
    return c;
}

std::unique_ptr<qwen3_5_model_state>
make_state(const metal_context& context, std::size_t blocks = 8)
{
    auto state = qwen3_5_model_state::make(context, small_config(), blocks, 2);
    REQUIRE(state);
    return std::move(*state);
}

void
fill(metal_tensor& tensor, float value)
{
    write_floats(tensor, std::vector<float>(tensor.descriptor().element_count(), value));
}

void
check_value(const metal_tensor& tensor, float value)
{
    const auto data = read_floats(tensor);
    CHECK(std::ranges::all_of(data, [=](float x) { return x == value; }));
}

model_batch
batch_for(const qwen3_5_model_state& state,
          std::initializer_list<seq_id> ids,
          std::size_t count = 1)
{
    model_batch batch { .id = 1,
                        .phase = batch_phase::prefill,
                        .kv_block_size = state.block_size(),
                        .tokens = {},
                        .positions = {},
                        .items = {} };
    for (auto id : ids) {
        const auto offset = batch.tokens.size();
        const auto start = state.committed_tokens(id).value_or(0);
        const auto blocks = state.resources(id).blocks;
        batch.items.push_back(
            { id, offset, count, offset + count - 1, { blocks.begin(), blocks.end() } });
        for (std::size_t i = 0; i < count; ++i) {
            batch.tokens.push_back(1);
            batch.positions.push_back(static_cast<position_id>(start + i));
        }
    }
    return batch;
}

// Run synchronous GPU writes to both types of state, finishing before the
// scheduler commits or aborts. Deterministic inputs depend on token position.
std::vector<float>
execute(const metal_context& context, qwen3_5_model_state& state, const model_batch& batch)
{
    auto weight = make_tensor(context, dtype::bf16, { 6, 1, 4 });
    write_bf16(weight, std::vector<float>(24, 0.25F));
    auto logs = make_tensor(context, dtype::f32, { 1 });
    fill(logs, -1);
    auto bias = make_tensor(context, dtype::bf16, { 1 });
    write_bf16(bias, { 0.5F });
    std::vector<float> outputs;
    for (const auto& item : batch.items) {
        const auto n = item.token_count;
        auto input = make_tensor(context, dtype::f32, { n, 6 });
        std::vector<float> data(n * 6);
        for (std::size_t t = 0; t < n; ++t)
            for (std::size_t f = 0; f < 6; ++f)
                data[t * 6 + f] = float(batch.positions[item.token_offset + t] + f + 1) / 10;
        write_floats(input, data);
        auto a = make_tensor(context, dtype::f32, { n, 1 });
        fill(a, 0.1F);
        auto b = make_tensor(context, dtype::f32, { n, 1 });
        fill(b, -0.2F);
        auto conv = make_tensor(context, dtype::f32, { n, 6 });
        auto output = make_tensor(context, dtype::f32, { n, 2 });
        for (const std::size_t layer : { 0, 2 }) {
            auto* memory = state.linear_state(item.id, layer);
            REQUIRE(memory);
            REQUIRE(causal_conv1d_silu(context, input, weight, memory->convolution, conv));
            REQUIRE(
                gated_delta_rule(context, conv, a, b, logs, bias, 1, memory->recurrent, output));
            const auto values = read_floats(output);
            outputs.insert(outputs.end(), values.begin(), values.end());
        }
    }
    auto metadata = prepare_paged_batch(batch, 32, state.cache().block_count(), state.block_size());
    REQUIRE(metadata);
    auto kv = make_tensor(context, dtype::f32, { batch.tokens.size(), 2 });
    std::vector<float> kv_data(batch.tokens.size() * 2);
    for (std::size_t row = 0; row < batch.tokens.size(); ++row)
        kv_data[2 * row] = kv_data[2 * row + 1] = float(batch.positions[row] + 1);
    write_floats(kv, kv_data);
    auto slots = upload_u32(context, metadata->slots);
    REQUIRE(slots);
    for (const std::size_t layer : { 1, 3 })
        REQUIRE(store_kv(context, kv, kv, *slots, *state.cache_layer(layer), state.cache()));
    return outputs;
}
} // namespace

TEST_CASE("hybrid state maps official layers to 18 recurrent states and six KV layers")
{
    const auto& context = test_context();
    auto config = load_qwen3_5_config(QWEN3_5_CONFIG_FIXTURE_PATH);
    REQUIRE(config);
    auto made = qwen3_5_model_state::make(context, *config, 1, 2);
    REQUIRE(made);
    auto& state = **made;
    CHECK(state.sequence_count() == 0);
    CHECK(state.linear_layer_count() == 18);
    CHECK(state.cache().layer_count() == 6);
    REQUIRE(state.reserve(7, 1));
    for (std::size_t layer = 0; layer < 24; ++layer) {
        if ((layer + 1) % 4 == 0) {
            CHECK(state.cache_layer(layer) == layer / 4);
            CHECK(state.linear_state(7, layer) == nullptr);
        } else {
            CHECK_FALSE(state.cache_layer(layer));
            const auto* memory = state.linear_state(7, layer);
            REQUIRE(memory);
            CHECK(std::ranges::equal(memory->convolution.descriptor().shape().dimensions(),
                                     std::vector<std::size_t> { 6144, 4 }));
            CHECK(std::ranges::equal(memory->recurrent.descriptor().shape().dimensions(),
                                     std::vector<std::size_t> { 16, 128, 128 }));
            check_value(memory->convolution, 0);
            check_value(memory->recurrent, 0);
        }
    }
    CHECK_FALSE(state.cache_layer(24));
    CHECK(state.linear_state(8, 0) == nullptr);
    CHECK(state.linear_state(7, 24) == nullptr);
}

TEST_CASE("hybrid reservations grow without resetting and release returns both memory and pages")
{
    auto state = make_state(test_context(), 3);
    CHECK(state->reserve(1, 0).error() == state_errc::invalid_reservation);
    CHECK(state->reserve(1, 33).error() == state_errc::invalid_reservation);
    REQUIRE(state->reserve(1, 2));
    auto* first = state->linear_state(1, 0);
    fill(first->convolution, 3);
    fill(first->recurrent, 4);
    REQUIRE(state->reserve(1, 2));
    REQUIRE(state->reserve(1, 3));
    CHECK(state->linear_state(1, 0) == first);
    check_value(first->convolution, 3);
    check_value(first->recurrent, 4);
    CHECK(state->resources(1).blocks.size() == 2);
    REQUIRE(state->reserve(2, 2));
    check_value(state->linear_state(2, 0)->recurrent, 0);
    check_value(state->linear_state(1, 2)->recurrent, 0);
    CHECK(state->reserve(3, 1).error() == state_errc::capacity_exhausted);
    CHECK(state->linear_state(3, 0) == nullptr);
    CHECK(state->resources(3).blocks.empty());
    CHECK(state->reserve(1, 5).error() == state_errc::capacity_exhausted);
    CHECK(state->resources(1).blocks.size() == 2);
    CHECK(state->reserve(1, 1).error() == state_errc::invalid_reservation);
    state->release(1);
    state->release(1);
    CHECK_FALSE(state->committed_tokens(1));
    CHECK(state->resources(1).blocks.empty());
    REQUIRE(state->reserve(1, 4));
    CHECK(state->committed_tokens(1) == 0);
    for (const std::size_t layer : { 0, 2 }) {
        check_value(state->linear_state(1, layer)->convolution, 0);
        check_value(state->linear_state(1, layer)->recurrent, 0);
    }
    CHECK(state->sequence_count() == 2);
}

TEST_CASE("hybrid transactions restore every participating layer and preserve committed memory")
{
    auto state = make_state(test_context());
    for (seq_id id : { 1, 2, 3 }) {
        REQUIRE(state->reserve(id, 4));
        for (const std::size_t layer : { 0, 2 }) {
            fill(state->linear_state(id, layer)->convolution, float(id + layer));
            fill(state->linear_state(id, layer)->recurrent, float(10 * id + layer));
        }
    }
    auto batch = batch_for(*state, { 1, 2 }, 2);
    REQUIRE(state->begin_batch(batch));
    CHECK(state->begin_batch(batch).error() == state_errc::invalid_reservation);
    CHECK(state->reserve(4, 1).error() == state_errc::invalid_reservation);
    for (seq_id id : { 1, 2 })
        for (const std::size_t layer : { 0, 2 }) {
            fill(state->linear_state(id, layer)->convolution, 99);
            fill(state->linear_state(id, layer)->recurrent, 98);
        }
    state->abort_batch();
    state->abort_batch();
    for (seq_id id : { 1, 2, 3 }) {
        CHECK(state->committed_tokens(id) == 0);
        for (const std::size_t layer : { 0, 2 }) {
            check_value(state->linear_state(id, layer)->convolution, float(id + layer));
            check_value(state->linear_state(id, layer)->recurrent, float(10 * id + layer));
        }
    }
    REQUIRE(state->begin_batch(batch));
    fill(state->linear_state(1, 0)->recurrent, 7);
    state->commit_batch();
    state->abort_batch();
    CHECK(state->committed_tokens(1) == 2);
    CHECK(state->committed_tokens(2) == 2);
    CHECK(state->committed_tokens(3) == 0);
    auto next = batch_for(*state, { 1 });
    REQUIRE(state->begin_batch(next));
    fill(state->linear_state(1, 0)->recurrent, 8);
    state->abort_batch();
    check_value(state->linear_state(1, 0)->recurrent, 7);
    CHECK(state->committed_tokens(1) == 2);
}

TEST_CASE("hybrid begin validates ownership and consecutive committed positions atomically")
{
    auto state = make_state(test_context());
    REQUIRE(state->reserve(1, 4));
    REQUIRE(state->reserve(2, 4));
    auto valid = batch_for(*state, { 1, 2 }, 2);
    auto invalid = [&](auto mutate) {
        auto batch = valid;
        mutate(batch);
        CHECK(state->begin_batch(batch).error() == state_errc::invalid_reservation);
        state->abort_batch();
        CHECK(state->committed_tokens(1) == 0);
        REQUIRE(state->begin_batch(valid));
        state->abort_batch();
    };
    invalid([](auto& b) { b.items[1].id = 99; });
    invalid([](auto& b) {
        b.items[1].id = 1;
        b.items[1].block_table = b.items[0].block_table;
    });
    invalid([](auto& b) { b.items[1].block_table = b.items[0].block_table; });
    invalid([](auto& b) { b.positions[0] = 1; });
    invalid([](auto& b) { b.positions[3] = 0; });
    invalid([](auto& b) { b.positions.pop_back(); });
    invalid([](auto& b) { b.items[1].token_offset = 0; });
    invalid([](auto& b) { b.kv_block_size = 3; });
    invalid([](auto& b) { b.phase = batch_phase::decode; });
    invalid([](auto& b) { b.items.clear(); });
    REQUIRE(state->begin_batch(valid));
    state->commit_batch();
    CHECK(state->begin_batch(valid).error() == state_errc::invalid_reservation);
    REQUIRE(state->reserve(3, 1));
    auto over_reserved = batch_for(*state, { 3 }, 2); // Fits page, exceeds reservation.
    CHECK(state->begin_batch(over_reserved).error() == state_errc::invalid_reservation);
}

TEST_CASE(
    "scheduler retries real DeltaNet mutations then commits partial prefill and releases on finish")
{
    const auto& context = test_context();
    auto storage = make_state(context);
    auto* state = storage.get();
    fill(state->cache().keys(), 0);
    fill(state->cache().values(), 0);
    scheduler_config config { .max_sequences = 2,
                              .max_batch_tokens = 2,
                              .kv_block_count = 8,
                              .kv_block_size = 2,
                              .eos_token = 99 };
    auto engine = scheduler::make(config, std::move(storage));
    REQUIRE(engine);
    auto sequence = seq::make(1, { 1, 2, 3 }, { .max_new_tokens = 2 });
    REQUIRE(sequence);
    REQUIRE(engine->add(std::move(*sequence)));
    auto scheduled = engine->schedule();
    REQUIRE(scheduled);
    auto batch = build_model_batch(*scheduled, *engine);
    REQUIRE(batch);
    REQUIRE(engine->begin_execution(*batch));
    const auto expected = execute(context, *state, *batch);
    const auto expected_conv = read_floats(state->linear_state(1, 0)->convolution);
    const auto expected_recurrent = read_floats(state->linear_state(1, 0)->recurrent);
    REQUIRE(engine->abort(scheduled->id));
    check_value(state->linear_state(1, 0)->convolution, 0);
    check_value(state->linear_state(1, 0)->recurrent, 0);
    CHECK(engine->find_sequence(1)->processed_token_count() == 0);
    scheduled = engine->schedule();
    REQUIRE(scheduled);
    batch = build_model_batch(*scheduled, *engine);
    REQUIRE(batch);
    REQUIRE(engine->begin_execution(*batch));
    CHECK(execute(context, *state, *batch) == expected);
    CHECK(read_floats(state->linear_state(1, 0)->convolution) == expected_conv);
    CHECK(read_floats(state->linear_state(1, 0)->recurrent) == expected_recurrent);
    REQUIRE(engine->complete(scheduled->id, {}));
    CHECK(state->committed_tokens(1) == 2);
    CHECK(engine->find_sequence(1)->processed_token_count() == 2);
    // Finish prefill, then one decode; the second sample hits max_new_tokens.
    for (int step = 0; step < 2; ++step) {
        scheduled = engine->schedule();
        REQUIRE(scheduled);
        batch = build_model_batch(*scheduled, *engine);
        REQUIRE(batch);
        const auto committed = *state->committed_tokens(1);
        const auto before_conv = read_floats(state->linear_state(1, 0)->convolution);
        const auto before_recurrent = read_floats(state->linear_state(1, 0)->recurrent);
        REQUIRE(engine->begin_execution(*batch));
        const auto attempted = execute(context, *state, *batch);
        REQUIRE(engine->abort(scheduled->id));
        CHECK(state->committed_tokens(1) == committed);
        CHECK(read_floats(state->linear_state(1, 0)->convolution) == before_conv);
        CHECK(read_floats(state->linear_state(1, 0)->recurrent) == before_recurrent);
        // An aborted append must never overwrite any committed KV prefix.
        const auto keys = read_floats(state->cache().keys());
        const auto vals = read_floats(state->cache().values());
        for (std::size_t layer = 0; layer < 2; ++layer)
            for (std::size_t pos = 0; pos < committed; ++pos)
                for (std::size_t feature = 0; feature < 2; ++feature) {
                    const auto block = state->resources(1).blocks[pos / 2];
                    const auto offset =
                        state->cache().element_offset(layer, block, pos % 2, 0, feature);
                    REQUIRE(offset);
                    CHECK(keys[*offset] == float(pos + 1));
                    CHECK(vals[*offset] == float(pos + 1));
                }
        scheduled = engine->schedule();
        REQUIRE(scheduled);
        batch = build_model_batch(*scheduled, *engine);
        REQUIRE(batch);
        REQUIRE(engine->begin_execution(*batch));
        CHECK(execute(context, *state, *batch) == attempted);
        const std::vector<token_id> samples { 42 };
        REQUIRE(engine->complete(scheduled->id, samples));
    }
    CHECK(state->sequence_count() == 0);
    CHECK(state->resources(1).blocks.empty());
    REQUIRE(engine->remove(1));
    sequence = seq::make(1, { 1 }, { .max_new_tokens = 4 });
    REQUIRE(sequence);
    REQUIRE(engine->add(std::move(*sequence)));
    scheduled = engine->schedule();
    REQUIRE(scheduled);
    check_value(state->linear_state(1, 0)->recurrent, 0);
    REQUIRE(engine->abort(scheduled->id)); // Abort before begin_execution is valid.
    REQUIRE(engine->cancel(1));
    CHECK(state->sequence_count() == 0);
}

TEST_CASE("hybrid geometry rejects zero, non-hybrid, and overflowing state layouts")
{
    const auto& context = test_context();
    auto invalid = [&](auto mutate) {
        auto config = small_config();
        mutate(config);
        CHECK(qwen3_5_model_state::make(context, config, 2, 2).error()
              == state_errc::invalid_reservation);
    };
    invalid([](auto& c) { c.layer_count = 0; });
    invalid([](auto& c) { c.layer_types.pop_back(); });
    invalid([](auto& c) { c.layer_types[0] = static_cast<qwen3_5_layer_type>(99); });
    invalid([](auto& c) {
        std::fill(c.layer_types.begin(), c.layer_types.end(), qwen3_5_layer_type::linear_attention);
    });
    invalid([](auto& c) { c.linear_key_head_count = 0; });
    invalid([](auto& c) { c.linear_key_head_count = 2; });
    invalid([](auto& c) { c.linear_key_head_dimension = std::numeric_limits<std::size_t>::max(); });
    invalid(
        [](auto& c) { c.linear_conv_kernel_dimension = std::numeric_limits<std::size_t>::max(); });
    invalid([](auto& c) { c.kv_head_count = 0; });
    CHECK(qwen3_5_model_state::make(context, small_config(), 0, 2).error()
          == state_errc::invalid_reservation);
    CHECK(qwen3_5_model_state::make(context, small_config(), 2, 0).error()
          == state_errc::invalid_reservation);
}
