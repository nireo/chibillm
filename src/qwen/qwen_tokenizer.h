#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "result.h"
#include "seq.h"

namespace chibillm {

enum class qwen_tokenizer_errc : std::uint8_t {
    file_read_failed,
    invalid_json,
    invalid_vocabulary,
    invalid_merge,
    invalid_utf8,
    token_not_found,
    token_id_out_of_range,
};

class qwen_tokenizer {
public:
    [[nodiscard]] static result<qwen_tokenizer, qwen_tokenizer_errc>
    load(const std::filesystem::path& path);

    [[nodiscard]] result<std::vector<token_id>, qwen_tokenizer_errc>
    encode(std::string_view text) const;

    [[nodiscard]] result<std::string, qwen_tokenizer_errc>
    decode(std::span<const token_id> tokens, bool skip_special_tokens = true) const;

private:
    qwen_tokenizer() = default;

    [[nodiscard]] result<std::vector<token_id>, qwen_tokenizer_errc>
    encode_piece(std::string_view piece) const;

    std::unordered_map<std::string, token_id> vocabulary_;
    std::vector<std::string> tokens_;
    std::vector<bool> special_tokens_;
    std::vector<std::pair<std::string, token_id>> added_tokens_;
    std::unordered_map<std::string, std::size_t> merge_ranks_;
    mutable std::unordered_map<std::string, std::vector<token_id>> piece_cache_;
    std::array<std::string, 256> byte_encoder_;
    std::unordered_map<char32_t, std::uint8_t> byte_decoder_;
};

} // namespace chibillm
