# chibillm

A small C++23 project for experimenting with LLM inference and Metal. This project implements enough LLM infra from scratch using C++ and Metal kernels to perform everything without dependencies. Probably going to add dependencies for things that don't really matter for this, such as a JSON parson etc.

## Build and run

Requires Meson and Ninja.

```sh
make build
make run
```

Run the tests with:

```sh
make test
```
