SHELL := /bin/sh

BUILD_DIR ?= build
MESON ?= meson
TEST ?=

.DEFAULT_GOAL := help

.PHONY: help setup reconfigure build test run format format-check clean

help:
	@printf '%s\n' \
		'make setup         configure the meson build directory' \
		'make reconfigure   force meson to reconfigure the project' \
		'make build         build every target' \
		'make test          build and run every registered test' \
		'make test TEST=x   build and run the named test' \
		'make run           build and run chibillm' \
		'make format        format source and test files' \
		'make format-check  check formatting without changing files' \
		'make clean         remove compiled outputs but keep configuration'

$(BUILD_DIR)/build.ninja: meson.build
	@if [ -d "$(BUILD_DIR)/meson-private" ]; then \
		$(MESON) setup --reconfigure "$(BUILD_DIR)" .; \
	else \
		$(MESON) setup "$(BUILD_DIR)" .; \
	fi

setup: $(BUILD_DIR)/build.ninja

reconfigure:
	@if [ -d "$(BUILD_DIR)/meson-private" ]; then \
		$(MESON) setup --reconfigure "$(BUILD_DIR)" .; \
	else \
		$(MESON) setup "$(BUILD_DIR)" .; \
	fi

build: setup
	$(MESON) compile -C "$(BUILD_DIR)"

test: build
	$(MESON) test -C "$(BUILD_DIR)" --print-errorlogs $(if $(strip $(TEST)),"$(TEST)")

run: build
	"$(BUILD_DIR)/chibillm"

format:
	./scripts/format.sh format

format-check:
	./scripts/format.sh check

clean: setup
	$(MESON) compile -C "$(BUILD_DIR)" --clean
