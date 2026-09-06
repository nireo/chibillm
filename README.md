# chibillm

A small C++23 inference engine for experimenting with Qwen inference on Metal. The model execution, paged KV cache, attention, tokenizer, scheduler, and generation loop are implemented in C++ and Metal. Currently the only model supported is Qwen3-0.6B, but I will try to implement different architectures for learning.

## Build and run

Requires Meson and Ninja.

```sh
make build
make run
```

`make run` loads the model from `qwen_model/` and starts a terminal chat. Pass another model directory directly when needed:

```sh
build/chibillm /path/to/qwen-model
```

## OpenAI-compatible server

Start the same model as an HTTP server with:

```sh
build/chibillm --serve /path/to/qwen-model
```

The server listens on `127.0.0.1:8000` and provides `GET /v1/models`,
`GET /v1/models/{model}`, and `POST /v1/chat/completions`. Chat completions accept
text messages, `max_completion_tokens` (or `max_tokens`), and `stream`. Several
requests can be active together and are batched by the inference scheduler.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_completion_tokens": 64
  }'
```

Set `stream` to `true` for a server-sent event stream ending in `data: [DONE]`.

Run the tests with:

```sh
make test
```

## Code structure

- `model_factory` selects a runner from the checkpoint's `model_type`.
- `inference_engine` executes scheduler reservations and publishes completion updates.
- `model_state` owns per-engine resources and provides batch begin/commit/abort and sequence release. Paged Metal state combines the block allocator and KV buffers; other architectures can provide their own state without changing sequence progress.
- `tensor/` contains reusable embedding, normalization, SwiGLU, attention, and fused greedy output operations. Attention metadata is prepared once per batch and its input tensors remain alive through the forward pass.
- `metal/` separates device/pass ownership, resource pooling, and kernel encoding.
- `model_format/weight_reader` validates and loads the same weight layouts, including packed projections.
- `serving_runtime` owns request execution; `server` maps requests and events to HTTP/JSON/SSE. Each request has a model-provided incremental text decoder.

To add an architecture, implement its runner, weight layout, and state, then add its factory entry. State that is mutated during execution must restore its pre-batch value on abort. Chat formatting and incremental decoding are separate from the forward pass. Qwen3.5 configuration, weight loading, hybrid state, and DeltaNet tensor primitives are available; its model forward pass is not implemented yet. `tensor/deltanet.h` provides stateful causal convolution with SiLU, the recurrent gated delta rule (including Q/K normalization and gate computation), and gated RMSNorm. The kernels process one sequence chunk per call with caller-owned FP32 state and support both prefill and decode; prefill currently uses a sequential scan. `qwen/qwen3_5_model_state.h` owns zero-initialized per-sequence convolution/recurrent memory and a compact KV cache for full-attention layers. It snapshots participating sequences at batch start, restores DeltaNet memory on abort, and tracks committed positions so retries overwrite only uncommitted KV slots. GPU work must finish before the scheduler commits, aborts, or releases state.

## Basic benchmark

Use the same release build, checkpoint, and machine for both runs. Save the old executable before rebuilding, then run:

```sh
python3 scripts/benchmark.py --binary /tmp/chibillm-before --output /tmp/before.json
python3 scripts/benchmark.py --binary build/chibillm --output /tmp/after.json
```

The script uses fixed short and long prompts, excludes one warmup per prompt from the medians, and records output hashes alongside prefill/decode rates and latency. CLI timing is rounded, so token rates are more precise than the printed first-token time.
