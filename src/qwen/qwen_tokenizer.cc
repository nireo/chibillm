#include "qwen/qwen_tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <iterator>
#include <limits>
#include <utility>

namespace chibillm {
namespace {

using json = nlohmann::json;

struct utf8_unit {
    char32_t codepoint;
    std::string_view bytes;
};

std::string
encode_utf8(char32_t codepoint)
{
    std::string output;
    if (codepoint <= 0x7F) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        output.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        output.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        output.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
    return output;
}

result<std::vector<utf8_unit>, qwen_tokenizer_errc>
split_utf8(std::string_view text)
{
    std::vector<utf8_unit> units;
    units.reserve(text.size());
    for (std::size_t offset = 0; offset < text.size();) {
        const auto first = static_cast<std::uint8_t>(text[offset]);
        std::size_t length = 0;
        char32_t codepoint = 0;
        if (first <= 0x7F) {
            length = 1;
            codepoint = first;
        } else if ((first & 0xE0) == 0xC0) {
            length = 2;
            codepoint = first & 0x1F;
        } else if ((first & 0xF0) == 0xE0) {
            length = 3;
            codepoint = first & 0x0F;
        } else if ((first & 0xF8) == 0xF0) {
            length = 4;
            codepoint = first & 0x07;
        } else {
            return fail(qwen_tokenizer_errc::invalid_utf8);
        }
        if (length > text.size() - offset) {
            return fail(qwen_tokenizer_errc::invalid_utf8);
        }
        for (std::size_t index = 1; index < length; ++index) {
            const auto continuation = static_cast<std::uint8_t>(text[offset + index]);
            if ((continuation & 0xC0) != 0x80) {
                return fail(qwen_tokenizer_errc::invalid_utf8);
            }
            codepoint = (codepoint << 6) | (continuation & 0x3F);
        }
        const auto minimum = length == 1 ? 0U
            : length == 2                ? 0x80U
            : length == 3                ? 0x800U
                                         : 0x10000U;
        if (codepoint < minimum
            || codepoint > 0x10FFFF
            || (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
            return fail(qwen_tokenizer_errc::invalid_utf8);
        }
        units.push_back({ codepoint, text.substr(offset, length) });
        offset += length;
    }
    return units;
}

bool
is_unicode_space(char32_t codepoint)
{
    return codepoint == 0x00A0
        || codepoint == 0x1680
        || (codepoint >= 0x2000 && codepoint <= 0x200A)
        || codepoint == 0x2028
        || codepoint == 0x2029
        || codepoint == 0x202F
        || codepoint == 0x205F
        || codepoint == 0x3000;
}

bool
is_letter(char32_t codepoint)
{
    if (codepoint < 128) {
        return std::isalpha(static_cast<unsigned char>(codepoint)) != 0;
    }
    // Qwen's regex uses Unicode letter classes. Treating non-ASCII codepoints
    // as letters preserves the important word boundaries without a Unicode dependency.
    return !is_unicode_space(codepoint);
}

bool
is_number(char32_t codepoint)
{
    return codepoint < 128 && std::isdigit(static_cast<unsigned char>(codepoint)) != 0;
}

bool
is_space(char32_t codepoint)
{
    if (codepoint < 128) {
        return std::isspace(static_cast<unsigned char>(codepoint)) != 0;
    }
    return is_unicode_space(codepoint);
}

bool
is_newline(char32_t codepoint)
{
    return codepoint == '\r' || codepoint == '\n';
}

std::string
join_units(const std::vector<utf8_unit>& units, std::size_t begin, std::size_t end)
{
    std::string output;
    for (std::size_t index = begin; index < end; ++index) {
        output.append(units[index].bytes);
    }
    return output;
}

bool
starts_contraction(const std::vector<utf8_unit>& units, std::size_t offset, std::size_t& length)
{
    if (units[offset].codepoint != '\'') {
        return false;
    }
    static constexpr std::string_view suffixes[] { "s", "t", "re", "ve", "m", "ll", "d" };
    for (const auto suffix : suffixes) {
        if (offset + 1 + suffix.size() > units.size()) {
            continue;
        }
        bool matches = true;
        for (std::size_t index = 0; index < suffix.size(); ++index) {
            const auto codepoint = units[offset + 1 + index].codepoint;
            if (codepoint >= 128
                || std::tolower(static_cast<unsigned char>(codepoint)) != suffix[index]) {
                matches = false;
                break;
            }
        }
        if (matches) {
            length = suffix.size() + 1;
            return true;
        }
    }
    return false;
}

result<std::vector<std::string>, qwen_tokenizer_errc>
pretokenize(std::string_view text)
{
    auto split = split_utf8(text);
    if (!split) {
        return fail(split.error());
    }
    const auto& units = *split;
    std::vector<std::string> pieces;
    for (std::size_t offset = 0; offset < units.size();) {
        std::size_t contraction_length = 0;
        if (starts_contraction(units, offset, contraction_length)) {
            pieces.push_back(join_units(units, offset, offset + contraction_length));
            offset += contraction_length;
            continue;
        }

        std::size_t end = offset;
        const bool prefixed_letter = !is_newline(units[offset].codepoint)
            && !is_letter(units[offset].codepoint)
            && !is_number(units[offset].codepoint)
            && offset + 1 < units.size()
            && is_letter(units[offset + 1].codepoint);
        if (is_letter(units[offset].codepoint) || prefixed_letter) {
            end += prefixed_letter ? 1 : 0;
            while (end < units.size() && is_letter(units[end].codepoint)) {
                ++end;
            }
        } else if (is_number(units[offset].codepoint)) {
            end = offset + 1;
        } else if (is_newline(units[offset].codepoint)) {
            end = offset + 1;
            while (end < units.size()
                   && (is_space(units[end].codepoint) || is_newline(units[end].codepoint))) {
                ++end;
            }
        } else if (is_space(units[offset].codepoint)) {
            end = offset + 1;
            if (units[offset].codepoint == ' '
                && end < units.size()
                && !is_space(units[end].codepoint)
                && !is_letter(units[end].codepoint)
                && !is_number(units[end].codepoint)) {
                while (end < units.size()
                       && !is_space(units[end].codepoint)
                       && !is_letter(units[end].codepoint)
                       && !is_number(units[end].codepoint)) {
                    ++end;
                }
            } else {
                while (end < units.size()
                       && is_space(units[end].codepoint)
                       && !is_newline(units[end].codepoint)) {
                    ++end;
                }
            }
        } else {
            end = offset + 1;
            while (end < units.size()
                   && !is_space(units[end].codepoint)
                   && !is_letter(units[end].codepoint)
                   && !is_number(units[end].codepoint)) {
                ++end;
            }
            while (end < units.size() && is_newline(units[end].codepoint)) {
                ++end;
            }
        }
        pieces.push_back(join_units(units, offset, end));
        offset = end;
    }
    return pieces;
}

std::string
merge_key(std::string_view first, std::string_view second)
{
    std::string key;
    key.reserve(first.size() + second.size() + 1);
    key.append(first);
    key.push_back('\0');
    key.append(second);
    return key;
}

std::pair<std::array<std::string, 256>, std::unordered_map<char32_t, std::uint8_t>>
make_byte_codec()
{
    std::vector<std::uint16_t> bytes;
    for (std::uint16_t value = '!'; value <= '~'; ++value)
        bytes.push_back(value);
    for (std::uint16_t value = 0xA1; value <= 0xAC; ++value)
        bytes.push_back(value);
    for (std::uint16_t value = 0xAE; value <= 0xFF; ++value)
        bytes.push_back(value);

    auto codepoints = bytes;
    std::uint16_t extra = 0;
    for (std::uint16_t value = 0; value <= 0xFF; ++value) {
        if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) {
            bytes.push_back(value);
            codepoints.push_back(static_cast<std::uint16_t>(256 + extra));
            ++extra;
        }
    }

    std::array<std::string, 256> encoder;
    std::unordered_map<char32_t, std::uint8_t> decoder;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        encoder[bytes[index]] = encode_utf8(codepoints[index]);
        decoder.emplace(codepoints[index], static_cast<std::uint8_t>(bytes[index]));
    }
    return { std::move(encoder), std::move(decoder) };
}

result<json, qwen_tokenizer_errc>
load_json(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return fail(qwen_tokenizer_errc::file_read_failed);
    }
    const std::string text { std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>() };
    if (input.bad()) {
        return fail(qwen_tokenizer_errc::file_read_failed);
    }
    auto value = json::parse(text, nullptr, false);
    if (value.is_discarded()) {
        return fail(qwen_tokenizer_errc::invalid_json);
    }
    return value;
}

} // namespace

result<qwen_tokenizer, qwen_tokenizer_errc>
qwen_tokenizer::load(const std::filesystem::path& model_directory)
{
    auto vocab = load_json(model_directory / "vocab.json");
    if (!vocab) {
        return fail(vocab.error());
    }
    if (!vocab->is_object()) {
        return fail(qwen_tokenizer_errc::invalid_vocabulary);
    }

    std::unordered_map<std::string, token_id> vocabulary;
    std::size_t largest_id = 0;
    for (const auto& [token, value] : vocab->items()) {
        if (!value.is_number_integer()) {
            return fail(qwen_tokenizer_errc::invalid_vocabulary);
        }
        const auto id = value.get<std::int64_t>();
        if (id < 0 || id > std::numeric_limits<token_id>::max()) {
            return fail(qwen_tokenizer_errc::invalid_vocabulary);
        }
        vocabulary.emplace(token, static_cast<token_id>(id));
        largest_id = std::max(largest_id, static_cast<std::size_t>(id));
    }

    auto tokenizer_config = load_json(model_directory / "tokenizer_config.json");
    if (!tokenizer_config) {
        return fail(tokenizer_config.error());
    }
    if (!tokenizer_config->is_object()) {
        return fail(qwen_tokenizer_errc::invalid_vocabulary);
    }
    const auto added = tokenizer_config->find("added_tokens_decoder");
    std::vector<std::pair<std::string, token_id>> added_tokens;
    std::vector<token_id> added_special_tokens;
    if (added != tokenizer_config->end()) {
        if (!added->is_object()) {
            return fail(qwen_tokenizer_errc::invalid_vocabulary);
        }
        for (const auto& [id_text, token] : added->items()) {
            std::int64_t id = 0;
            const auto parsed =
                std::from_chars(id_text.data(), id_text.data() + id_text.size(), id);
            if (parsed.ec != std::errc {}
                || parsed.ptr != id_text.data() + id_text.size()
                || id < 0
                || id > std::numeric_limits<token_id>::max()
                || !token.is_object()
                || !token.contains("content")
                || !token["content"].is_string()) {
                return fail(qwen_tokenizer_errc::invalid_vocabulary);
            }
            auto content = token["content"].get<std::string>();
            vocabulary[content] = static_cast<token_id>(id);
            added_tokens.emplace_back(std::move(content), static_cast<token_id>(id));
            if (token.value("special", false)) {
                added_special_tokens.push_back(static_cast<token_id>(id));
            }
            largest_id = std::max(largest_id, static_cast<std::size_t>(id));
        }
    }

    std::vector<std::string> tokens(largest_id + 1);
    std::vector<bool> special_tokens(largest_id + 1, false);
    for (const auto& [token, id] : vocabulary) {
        tokens[static_cast<std::size_t>(id)] = token;
    }
    for (const auto id : added_special_tokens) {
        special_tokens[static_cast<std::size_t>(id)] = true;
    }

    std::ifstream merges(model_directory / "merges.txt");
    if (!merges) {
        return fail(qwen_tokenizer_errc::file_read_failed);
    }
    std::unordered_map<std::string, std::size_t> merge_ranks;
    merge_ranks.reserve(vocabulary.size());
    std::string line;
    if (!std::getline(merges, line) || !line.starts_with("#version:")) {
        return fail(qwen_tokenizer_errc::invalid_merge);
    }
    std::size_t rank = 0;
    while (std::getline(merges, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto separator = line.find(' ');
        if (separator == std::string::npos || separator == 0 || separator + 1 >= line.size()) {
            return fail(qwen_tokenizer_errc::invalid_merge);
        }
        merge_ranks.emplace(merge_key(std::string_view(line).substr(0, separator),
                                      std::string_view(line).substr(separator + 1)),
                            rank++);
    }
    if (merges.bad()) {
        return fail(qwen_tokenizer_errc::file_read_failed);
    }

    std::ranges::sort(added_tokens, [](const auto& first, const auto& second) {
        return first.first.size() > second.first.size();
    });
    qwen_tokenizer tokenizer;
    tokenizer.vocabulary_ = std::move(vocabulary);
    tokenizer.tokens_ = std::move(tokens);
    tokenizer.special_tokens_ = std::move(special_tokens);
    tokenizer.added_tokens_ = std::move(added_tokens);
    tokenizer.merge_ranks_ = std::move(merge_ranks);
    auto [encoder, decoder] = make_byte_codec();
    tokenizer.byte_encoder_ = std::move(encoder);
    tokenizer.byte_decoder_ = std::move(decoder);
    return tokenizer;
}

result<std::vector<token_id>, qwen_tokenizer_errc>
qwen_tokenizer::encode_piece(std::string_view piece) const
{
    const auto cached = piece_cache_.find(std::string(piece));
    if (cached != piece_cache_.end()) {
        return cached->second;
    }

    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (const auto byte : piece) {
        symbols.push_back(byte_encoder_[static_cast<std::uint8_t>(byte)]);
    }

    while (symbols.size() > 1) {
        auto best_rank = std::numeric_limits<std::size_t>::max();
        std::string best_first;
        std::string best_second;
        for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
            const auto found = merge_ranks_.find(merge_key(symbols[index], symbols[index + 1]));
            if (found != merge_ranks_.end() && found->second < best_rank) {
                best_rank = found->second;
                best_first = symbols[index];
                best_second = symbols[index + 1];
            }
        }
        if (best_rank == std::numeric_limits<std::size_t>::max()) {
            break;
        }

        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (std::size_t index = 0; index < symbols.size();) {
            if (index + 1 < symbols.size()
                && symbols[index] == best_first
                && symbols[index + 1] == best_second) {
                merged.push_back(symbols[index] + symbols[index + 1]);
                index += 2;
            } else {
                merged.push_back(std::move(symbols[index]));
                ++index;
            }
        }
        symbols = std::move(merged);
    }

    std::vector<token_id> output;
    output.reserve(symbols.size());
    for (const auto& symbol : symbols) {
        const auto found = vocabulary_.find(symbol);
        if (found == vocabulary_.end()) {
            return fail(qwen_tokenizer_errc::token_not_found);
        }
        output.push_back(found->second);
    }
    piece_cache_.emplace(std::string(piece), output);
    return output;
}

result<std::vector<token_id>, qwen_tokenizer_errc>
qwen_tokenizer::encode(std::string_view text) const
{
    std::vector<token_id> output;
    const auto encode_ordinary =
        [&](std::string_view ordinary) -> result<void, qwen_tokenizer_errc> {
        auto pieces = pretokenize(ordinary);
        if (!pieces) {
            return fail(pieces.error());
        }
        for (const auto& piece : *pieces) {
            auto encoded = encode_piece(piece);
            if (!encoded) {
                return fail(encoded.error());
            }
            output.insert(output.end(), encoded->begin(), encoded->end());
        }
        return {};
    };

    std::size_t offset = 0;
    while (offset < text.size()) {
        auto next_offset = std::string_view::npos;
        const std::pair<std::string, token_id>* next_token = nullptr;
        for (const auto& token : added_tokens_) {
            const auto found = text.find(token.first, offset);
            if (found < next_offset) {
                next_offset = found;
                next_token = &token;
            }
        }
        if (next_token == nullptr) {
            auto encoded = encode_ordinary(text.substr(offset));
            if (!encoded) {
                return fail(encoded.error());
            }
            break;
        }
        auto encoded = encode_ordinary(text.substr(offset, next_offset - offset));
        if (!encoded) {
            return fail(encoded.error());
        }
        output.push_back(next_token->second);
        offset = next_offset + next_token->first.size();
    }
    return output;
}

result<std::string, qwen_tokenizer_errc>
qwen_tokenizer::decode(std::span<const token_id> tokens, bool skip_special_tokens) const
{
    std::string output;
    for (const auto id : tokens) {
        if (id < 0
            || static_cast<std::size_t>(id) >= tokens_.size()
            || tokens_[static_cast<std::size_t>(id)].empty()) {
            return fail(qwen_tokenizer_errc::token_id_out_of_range);
        }
        const auto index = static_cast<std::size_t>(id);
        if (special_tokens_[index]) {
            if (!skip_special_tokens) {
                output += tokens_[index];
            }
            continue;
        }
        auto units = split_utf8(tokens_[index]);
        if (!units) {
            return fail(units.error());
        }
        for (const auto& unit : *units) {
            const auto found = byte_decoder_.find(unit.codepoint);
            if (found == byte_decoder_.end()) {
                return fail(qwen_tokenizer_errc::invalid_vocabulary);
            }
            output.push_back(static_cast<char>(found->second));
        }
    }
    return output;
}

} // namespace chibillm
