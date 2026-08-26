#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "qwen/qwen_tokenizer.h"

namespace {

class tokenizer_fixture {
public:
    tokenizer_fixture()
        : path_(std::filesystem::temp_directory_path() / "chibillm_tokenizer_test")
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
        REQUIRE(std::filesystem::create_directory(path_));

        const nlohmann::json vocabulary {
            { "!", 0 },     { "H", 1 },      { "e", 2 },       { "l", 3 },     { "o", 4 },
            { "He", 5 },    { "Hel", 6 },    { "Hell", 7 },    { "Hello", 8 }, { "Ġ", 9 },
            { "w", 10 },    { "r", 11 },     { "d", 12 },      { "wo", 13 },   { "wor", 14 },
            { "worl", 15 }, { "world", 16 }, { "Ġworld", 17 }, { "Ċ", 18 },
        };
        std::ofstream vocab_output(path_ / "vocab.json");
        vocab_output << vocabulary;
        REQUIRE(vocab_output.good());

        std::ofstream merges_output(path_ / "merges.txt");
        merges_output
            << "#version: 0.2\nH e\nHe l\nHel l\nHell o\nw o\nwo r\nwor l\nworl d\nĠ world\n";
        REQUIRE(merges_output.good());

        const nlohmann::json added_tokens {
            { "19", { { "content", "<|im_start|>" }, { "special", true } } },
            { "20", { { "content", "<|im_end|>" }, { "special", true } } },
            { "21", { { "content", "<think>" }, { "special", false } } },
        };
        std::ofstream config_output(path_ / "tokenizer_config.json");
        config_output << nlohmann::json { { "added_tokens_decoder", added_tokens } };
        REQUIRE(config_output.good());
    }

    ~tokenizer_fixture()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path&
    path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

} // namespace

TEST_CASE("Qwen byte-level BPE encodes and decodes text")
{
    tokenizer_fixture fixture;
    auto tokenizer = chibillm::qwen_tokenizer::load(fixture.path());
    CAPTURE(tokenizer.has_value() ? -1 : static_cast<int>(tokenizer.error()));
    REQUIRE(tokenizer.has_value());

    auto encoded = tokenizer->encode("Hello world!\n");
    REQUIRE(encoded.has_value());
    CHECK(*encoded == std::vector<chibillm::token_id> { 8, 17, 0, 18 });

    auto decoded = tokenizer->decode(*encoded);
    REQUIRE(decoded.has_value());
    CHECK(*decoded == "Hello world!\n");
}

TEST_CASE("Qwen tokenizer exposes and skips chat special tokens")
{
    tokenizer_fixture fixture;
    auto tokenizer = chibillm::qwen_tokenizer::load(fixture.path());
    CAPTURE(tokenizer.has_value() ? -1 : static_cast<int>(tokenizer.error()));
    REQUIRE(tokenizer.has_value());

    auto chat = tokenizer->encode("<|im_start|>Hello<|im_end|>");
    REQUIRE(chat.has_value());
    CHECK(*chat == std::vector<chibillm::token_id> { 19, 8, 20 });
    CHECK(tokenizer->decode(*chat).value() == "Hello");
    CHECK(tokenizer->decode(*chat, false).value() == "<|im_start|>Hello<|im_end|>");

    auto thinking = tokenizer->encode("<think>");
    REQUIRE(thinking.has_value());
    CHECK(*thinking == std::vector<chibillm::token_id> { 21 });
    CHECK(tokenizer->decode(*thinking).value() == "<think>");
}
