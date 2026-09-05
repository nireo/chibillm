#include "model_runner.h"
#include "text.h"
#include <optional>
#include <string_view>

namespace chibillm {
[[nodiscard]] std::optional<std::size_t>
complete_utf8_prefix(std::string_view text)
{
    std::size_t offset = 0;
    while (offset < text.size()) {
        const auto first = static_cast<unsigned char>(text[offset]);
        const std::size_t length = first < 0x80 ? 1
            : (first & 0xE0) == 0xC0            ? 2
            : (first & 0xF0) == 0xE0            ? 3
            : (first & 0xF8) == 0xF0            ? 4
                                                : 0;
        if (length == 0) {
            return std::nullopt;
        }
        if (length > text.size() - offset) {
            break;
        }
        char32_t codepoint = first
            & (length == 1       ? 0x7F
                   : length == 2 ? 0x1F
                   : length == 3 ? 0x0F
                                 : 0x07);
        for (std::size_t index = 1; index < length; ++index) {
            const auto byte = static_cast<unsigned char>(text[offset + index]);
            if ((byte & 0xC0) != 0x80)
                return std::nullopt;
            codepoint = (codepoint << 6) | (byte & 0x3F);
        }
        const auto minimum = length == 1 ? 0U
            : length == 2                ? 0x80U
            : length == 3                ? 0x800U
                                         : 0x10000U;
        if (codepoint < minimum
            || codepoint > 0x10FFFF
            || (codepoint >= 0xD800 && codepoint <= 0xDFFF))
            return std::nullopt;
        offset += length;
    }
    return offset;
}

namespace {
class prefix_decoder final : public text_decoder {
public:
    explicit prefix_decoder(const model_runner& runner)
        : runner_(runner)
    {}

    result<std::string, model_runner_errc>
    push(token_id token, bool final) override
    {
        tokens_.push_back(token);
        auto text = runner_.decode(tokens_);
        if (!text)
            return fail(text.error());
        const auto prefix = complete_utf8_prefix(*text);
        if (!prefix || (final && *prefix != text->size()) || !text->starts_with(emitted_)) {
            return fail(model_runner_errc::tokenizer_failure);
        }
        auto delta = text->substr(emitted_.size(), *prefix - emitted_.size());
        emitted_.append(delta);
        return delta;
    }

private:
    const model_runner& runner_;
    std::vector<token_id> tokens_;
    std::string emitted_;
};
} // namespace

std::unique_ptr<text_decoder>
model_runner::make_decoder() const
{
    return std::make_unique<prefix_decoder>(*this);
}
} // namespace chibillm
