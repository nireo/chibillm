# chibillm

A small C++23 inference engine for experimenting with Qwen inference on Metal. The model execution, paged KV cache, attention, tokenizer, scheduler, and generation loop are implemented in C++ and Metal. Supports Qwen3-0.6B and text generation with Qwen3.5-0.8B.

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

To use the local Qwen3.5 checkpoint:

```sh
build/chibillm qwen3_5_model
```

Qwen3.5 loading accepts `model.safetensors` or a single `.safetensors` file in the
model directory, including the official single-shard filename. Multiple shards
are not supported. This runner loads the text decoder with tied embeddings;
image/video inputs and multi-token prediction are not supported.

The app defaults to 32,768 tokens of context (prompt plus reply) and up to 8,192
tokens per reply. Both limits can be set at startup, in chat or server mode:

```sh
build/chibillm --context-length 65536 --max-tokens 16384 qwen3_5_model
build/chibillm --serve --context-length 65536 --max-tokens 16384 qwen3_5_model
```

Context length must be a positive multiple of 16 and is capped at the checkpoint's
configured maximum: 262,144 for Qwen3.5-0.8B, or 32,768 for Qwen3-0.6B. Chat replies
also stop at the remaining context capacity. In server mode, `--max-tokens` sets
the default; individual requests can override it with `max_completion_tokens`.
The server requires the prompt plus requested output budget to fit the context.

Qwen3.5's FP32 KV cache uses 768 MiB at 32K context, 1.5 GiB at 64K, 3 GiB at 128K,
and 6 GiB at 256K, in addition to approximately 1.4 GiB of text weights and runtime
memory. Server requests share this cache capacity. Longer contexts take more time
to process; increasing the limit does not force the model to produce longer replies.

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

- `main` handles argument errors and help, then starts the application. `cli_options` parses startup options; `application` loads the model and starts chat or server mode.
- `repl` owns terminal input, conversation history, generation, sequence cleanup, and performance reporting.
- `model_factory` selects a runner from the checkpoint's `model_type`.
- `inference_engine` executes scheduler reservations and publishes completion updates.
- `model_state` owns per-engine resources and provides batch begin/commit/abort and sequence release. Paged Metal state combines the block allocator and KV buffers; other architectures can provide their own state without changing sequence progress.
- `tensor/` contains reusable embedding, normalization, SwiGLU, attention, and fused greedy output operations. Attention metadata is prepared once per batch and its input tensors remain alive through the forward pass.
- `metal/` separates device/pass ownership, resource pooling, and kernel encoding.
- `model_format/weight_reader` validates and loads the same weight layouts, including packed projections.
- `serving_runtime` owns request execution; `server` maps requests and events to HTTP/JSON/SSE. Each request has a model-provided incremental text decoder.

To add an architecture, implement its runner, weight layout, and state, then add its factory entry. State that is mutated during execution must restore its pre-batch value on abort. Chat formatting and incremental decoding are separate from the forward pass.

Qwen3.5's runner executes the hybrid layer loop, applies zero-centered final RMSNorm,
and uses the tied embedding for greedy output. It supports chunked prefill, cached
decode, and multiple sequences through the same CLI and HTTP server as Qwen3.
`run_qwen3_5_full_attention` applies zero-centered input and Q/K norms, per-head
query/gate splitting, partial RoPE, paged attention, sigmoid output gating, and the
output projection with residual addition. `tensor/deltanet.h` provides stateful
causal convolution with SiLU, the recurrent gated delta rule, and gated RMSNorm.
The kernels process one sequence chunk per call with caller-owned FP32 state;
prefill currently uses a sequential scan.

`qwen/qwen3_5_model_state.h` owns zero-initialized per-sequence convolution/recurrent
memory and a compact KV cache for full-attention layers. It snapshots participating
sequences at batch start, restores DeltaNet memory on abort, and tracks committed
positions so retries overwrite only uncommitted KV slots. GPU work finishes before
the scheduler commits, aborts, or releases state. The tests also check generation
with the official Qwen3.5 checkpoint when it is installed in `qwen3_5_model/`.

## Basic benchmark

Use the same release build, checkpoint, and machine for both runs. Save the old executable before rebuilding, then run:

```sh
python3 scripts/benchmark.py --binary /tmp/chibillm-before --output /tmp/before.json
python3 scripts/benchmark.py --binary build/chibillm --output /tmp/after.json
```

The script uses fixed short and long prompts, excludes one warmup per prompt from the medians, and records output hashes alongside prefill/decode rates and latency. CLI timing is rounded, so token rates are more precise than the printed first-token time.
