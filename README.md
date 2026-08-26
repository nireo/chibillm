# chibillm

A small C++23 project for experimenting with Qwen inference on Metal. The model execution, paged KV cache, attention, tokenizer, scheduler, and generation loop are implemented in C++ and Metal. nlohmann/json is used for model metadata.

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

Inside the chat, use `/reset` to clear conversation history and `/quit` to exit.
After each response, a `[perf]` line reports time to first token, approximate prefill
throughput, decode throughput, and total generation time.

Run the tests with:

```sh
make test
```
