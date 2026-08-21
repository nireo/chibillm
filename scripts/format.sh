#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
mode=${1:-format}

case "$mode" in
format)
    find "$project_root/src" "$project_root/tests" "$project_root/shaders" \
        -type f \
        \( -name '*.h' -o -name '*.cc' -o -name '*.cpp' -o -name '*.mm' -o -name '*.metal' \) \
        -exec clang-format -i {} +
    ;;
check)
    find "$project_root/src" "$project_root/tests" "$project_root/shaders" \
        -type f \
        \( -name '*.h' -o -name '*.cc' -o -name '*.cpp' -o -name '*.mm' -o -name '*.metal' \) \
        -exec clang-format --dry-run --Werror {} +
    ;;
*)
    echo "usage: $0 [format|check]" >&2
    exit 2
    ;;
esac
