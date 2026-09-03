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
