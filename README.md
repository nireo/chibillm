# chibillm

chibillm is a small C++23 inference engine for text generation with Qwen models on Apple Metal. It supports Qwen3-0.6B and Qwen3.5-0.8B, with interactive terminal chat and an OpenAI-compatible HTTP server.

The engine includes model loading, tokenization, a generation loop, and a scheduler for batched inference. Metal kernels handle model computation, while the runtime manages attention and KV-cache state. The code is organized around model runners and reusable tensor operations, so model-specific execution is kept separate from shared inference and serving code.

Metal compute sources live in `src/metal/kernels/`, grouped into linear projections, attention, DeltaNet, sampling, and general tensor operations. Matching dispatch implementations live in `src/metal/metal_kernels_*.mm`. Meson bundles the sources into an embedded header, which the runtime compiles into one Metal library. Kernel edits automatically rebuild the embedded source; the executable does not need source files at runtime.

Requires Meson and Ninja. Build and run with a Qwen3.5 checkpoint in `qwen3_5_model/`, or pass another model directory to the executable. Run the test suite with `make test`.

```sh
make build
make run
```
