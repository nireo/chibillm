#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
mode=${1:-format}

case "$mode" in
format)
    clang-format -i \
        "$project_root"/src/*.h \
        "$project_root"/src/*.cc \
        "$project_root"/tests/*.cc
    ;;
check)
    clang-format --dry-run --Werror \
        "$project_root"/src/*.h \
        "$project_root"/src/*.cc \
        "$project_root"/tests/*.cc
    ;;
*)
    echo "usage: $0 [format|check]" >&2
    exit 2
    ;;
esac
