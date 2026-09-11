#include "cli_options.h"

#include <charconv>
#include <ostream>
#include <string_view>

namespace chibillm {

void
print_usage(std::ostream& output)
{
    output << "Usage: chibillm [--serve] [--context-length N] [--max-tokens N] [model-directory]\n"
              "  --context-length N  Prompt + output capacity (default: 32768; multiple of 16)\n"
              "  --max-tokens N      Reply limit / server default (default: 8192)\n"
              "  --serve             Start the HTTP server on 127.0.0.1:8000\n"
              "  --no-stream         Print REPL replies only when complete\n"
              "  --progress=auto|off  REPL prefill progress on terminals (default: auto)\n"
              "  --metrics-jsonl PATH  Append full-precision REPL request metrics (no text)\n"
              "  --help              Show this help\n";
}

result<cli_options, std::string>
parse_cli_options(int argc, char** argv)
{
    cli_options settings;
    bool has_directory = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            settings.help = true;
        } else if (arg == "--no-stream") {
            settings.stream = false;
        } else if (arg == "--progress=auto" || arg == "--progress=off") {
            settings.progress = arg == "--progress=auto";
        } else if (arg == "--metrics-jsonl") {
            if (++i >= argc
                || std::string_view(argv[i]).empty()
                || std::string_view(argv[i]).starts_with('-'))
                return fail("missing path for --metrics-jsonl");
            settings.metrics_jsonl = argv[i];
        } else if (arg == "--serve") {
            settings.serve = true;
        } else if (arg == "--context-length" || arg == "--max-tokens") {
            std::size_t value = 0;
            if (++i >= argc)
                return fail("missing value for " + std::string(arg));
            const std::string_view text(argv[i]);
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
            if (parsed.ec != std::errc {}
                || parsed.ptr != text.data() + text.size()
                || value == 0
                || (arg == "--context-length" && value % cli_options::kv_block_size != 0)) {
                return fail(
                    "invalid token limit for " + std::string(arg) + ": " + std::string(text));
            }
            if (arg == "--context-length")
                settings.context_length = value;
            else
                settings.max_tokens = value;
        } else if (arg.starts_with('-') || has_directory) {
            return fail("unexpected argument: " + std::string(arg));
        } else {
            settings.model_directory = arg;
            has_directory = true;
        }
    }
    if (settings.serve && !settings.metrics_jsonl.empty())
        return fail("--metrics-jsonl currently supports REPL mode only");
    return settings;
}

} // namespace chibillm
