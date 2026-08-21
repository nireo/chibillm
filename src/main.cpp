#include <iostream>
#include <utility>

#include "inference_engine.h"
#include "model_runner.h"

int main()
{
    using namespace chibillm;

    fake_model_runner runner { 42 };
    auto engine = inference_engine::make(
        scheduler_config {
            .max_sequences = 4,
            .max_batch_tokens = 2,
            .kv_block_count = 16,
            .kv_block_size = 2,
            .eos_token = 99,
        },
        runner);
    if (!engine) {
        std::cerr << "failed to create the inference engine\n";
        return 1;
    }

    auto sequence = seq::make(1, { 10, 20, 30 },
        sampling_params {
            .temperature = 1.0F,
            .max_new_tokens = 5,
            .ignore_eos = false,
        },
        2);
    if (!sequence || !engine->add(std::move(*sequence))) {
        std::cerr << "failed to add the demo sequence\n";
        return 1;
    }

    while (!engine->is_finished()) {
        if (!engine->step()) {
            std::cerr << "inference step failed\n";
            return 1;
        }
    }

    const auto* finished = engine->find_sequence(1);
    if (finished == nullptr) {
        std::cerr << "finished sequence is missing\n";
        return 1;
    }

    std::cout << "generated token ids:";
    for (const auto token : finished->completion_tokens()) {
        std::cout << ' ' << token;
    }
    std::cout << '\n';

    return 0;
}
