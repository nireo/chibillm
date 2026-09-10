#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

#include "model_format/safetensors.h"
#include "safetensors_test_support.h"

using chibillm::safetensors_dtype;
using chibillm::safetensors_errc;
using chibillm::safetensors_file;
using safetensors_test::temporary_file;

namespace {

class temporary_directory {
public:
    explicit temporary_directory(const char* name)
        : path_(std::filesystem::temp_directory_path() / name)
    {
        REQUIRE(std::filesystem::create_directory(path_));
    }

    temporary_directory(const temporary_directory&) = delete;
    temporary_directory& operator=(const temporary_directory&) = delete;

    ~temporary_directory()
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

TEST_CASE("safetensors exposes metadata and reads exact tensor bytes")
{
    nlohmann::json header;
    std::vector<std::byte> data;
    safetensors_test::add_tensor(header, data, "weight", "BF16", { 2, 2 });
    safetensors_test::add_tensor(header, data, "bias", "F32", { 2 });
    for (std::size_t index = 0; index < data.size(); ++index) {
        data[index] = static_cast<std::byte>(index);
    }
    temporary_file file("chibillm_safetensors_valid.safetensors", header, data);

    auto opened = safetensors_file::open(file.path());
    REQUIRE(opened.has_value());
    CHECK(opened->tensor_count() == 2);

    const auto* weight = opened->find("weight");
    REQUIRE(weight != nullptr);
    CHECK(weight->type == safetensors_dtype::bf16);
    CHECK(weight->shape == std::vector<std::size_t> { 2, 2 });
    CHECK(weight->byte_count == 8);

    std::vector<std::byte> weight_bytes(8);
    REQUIRE(opened->read("weight", weight_bytes).has_value());
    for (std::size_t index = 0; index < weight_bytes.size(); ++index) {
        CHECK(weight_bytes[index] == static_cast<std::byte>(index));
    }

    std::vector<std::byte> wrong_size(7);
    CHECK(opened->read("weight", wrong_size).error()
          == safetensors_errc::destination_size_mismatch);
    CHECK(opened->read("missing", wrong_size).error() == safetensors_errc::tensor_not_found);
}

TEST_CASE("safetensors model resolution prefers the canonical file")
{
    temporary_directory directory("chibillm_safetensors_model_canonical");
    nlohmann::json header;
    std::vector<std::byte> data;
    safetensors_test::add_tensor(header, data, "canonical", "BF16", { 1 });
    temporary_file canonical((directory.path() / "model.safetensors").string(), header, data);
    temporary_file first((directory.path() / "first.safetensors").string(),
                         nlohmann::json::object(), {});
    temporary_file second((directory.path() / "second.safetensors").string(),
                          nlohmann::json::object(), {});

    auto opened = safetensors_file::open_model(directory.path());
    REQUIRE(opened.has_value());
    CHECK(opened->tensor_count() == 1);
    CHECK(opened->find("canonical") != nullptr);
}

TEST_CASE("safetensors model resolution opens a sole noncanonical shard")
{
    temporary_directory directory("chibillm_safetensors_model_sole");
    nlohmann::json header;
    std::vector<std::byte> data;
    safetensors_test::add_tensor(header, data, "weight", "BF16", { 1 });
    data[0] = std::byte { 42 };
    temporary_file file(
        (directory.path() / "model.safetensors-00001-of-00001.safetensors").string(), header, data);
    REQUIRE(std::filesystem::create_directory(directory.path() / "ignored.safetensors"));
    std::ofstream(directory.path() / "model.safetensors.index.json") << "{}";

    auto opened = safetensors_file::open_model(directory.path());
    REQUIRE(opened.has_value());
    CHECK(opened->tensor_count() == 1);
    std::vector<std::byte> bytes(data.size());
    REQUIRE(opened->read("weight", bytes).has_value());
    CHECK(bytes == data);
}

TEST_CASE("safetensors model resolution rejects multiple fallback candidates")
{
    temporary_directory directory("chibillm_safetensors_model_ambiguous");
    temporary_file first((directory.path() / "model-00001-of-00002.safetensors").string(),
                         nlohmann::json::object(), {});
    temporary_file second((directory.path() / "model-00002-of-00002.safetensors").string(),
                          nlohmann::json::object(), {});

    auto opened = safetensors_file::open_model(directory.path());
    REQUIRE_FALSE(opened.has_value());
    CHECK(opened.error() == safetensors_errc::ambiguous_checkpoint);
}

TEST_CASE("safetensors model resolution rejects missing checkpoints")
{
    temporary_directory directory("chibillm_safetensors_model_missing");
    auto path = directory.path();

    SUBCASE("missing directory")
    {
        path /= "missing";
    }
    SUBCASE("empty directory") {}
    SUBCASE("no regular safetensors files")
    {
        REQUIRE(std::filesystem::create_directory(path / "nested.safetensors"));
        temporary_file nested((path / "nested.safetensors" / "weights.safetensors").string(),
                              nlohmann::json::object(), {});
        std::ofstream(path / "config.json") << "{}";
        auto opened = safetensors_file::open_model(path);
        REQUIRE_FALSE(opened.has_value());
        CHECK(opened.error() == safetensors_errc::file_open_failed);
    }

    auto opened = safetensors_file::open_model(path);
    REQUIRE_FALSE(opened.has_value());
    CHECK(opened.error() == safetensors_errc::file_open_failed);
}

TEST_CASE("safetensors model resolution returns filesystem errors")
{
    temporary_directory directory("chibillm_safetensors_model_filesystem_error");
    auto path = directory.path();

    SUBCASE("non-directory argument")
    {
        path /= "not_a_directory";
        std::ofstream(path) << "not a directory";
    }
    SUBCASE("canonical status error")
    {
        std::filesystem::create_symlink("model.safetensors", path / "model.safetensors");
    }
    SUBCASE("candidate status error")
    {
        std::filesystem::create_symlink("loop.safetensors", path / "loop.safetensors");
    }

    auto opened = safetensors_file::open_model(path);
    REQUIRE_FALSE(opened.has_value());
    CHECK(opened.error() == safetensors_errc::file_open_failed);
}

TEST_CASE("safetensors model resolution preserves selected file errors")
{
    temporary_directory directory("chibillm_safetensors_model_malformed");

    SUBCASE("malformed canonical file does not fall back to a valid alternative")
    {
        temporary_file canonical((directory.path() / "model.safetensors").string(),
                                 nlohmann::json::array(), {});
        temporary_file alternative((directory.path() / "weights.safetensors").string(),
                                   nlohmann::json::object(), {});
        auto opened = safetensors_file::open_model(directory.path());
        REQUIRE_FALSE(opened.has_value());
        CHECK(opened.error() == safetensors_errc::invalid_header_json);
    }
    SUBCASE("sole candidate with a truncated header")
    {
        const auto path = directory.path() / "weights.safetensors";
        std::ofstream(path, std::ios::binary) << "short";
        auto opened = safetensors_file::open_model(directory.path());
        REQUIRE_FALSE(opened.has_value());
        CHECK(opened.error() == safetensors_errc::invalid_header_size);
    }
}

TEST_CASE("safetensors rejects malformed headers")
{
    SUBCASE("missing header length")
    {
        const auto path =
            std::filesystem::temp_directory_path() / "chibillm_safetensors_short.safetensors";
        {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            output << "short";
        }
        CHECK(safetensors_file::open(path).error() == safetensors_errc::invalid_header_size);
        std::filesystem::remove(path);
    }

    SUBCASE("non-object JSON header")
    {
        temporary_file file("chibillm_safetensors_bad_json.safetensors", nlohmann::json::array(),
                            {});
        CHECK(safetensors_file::open(file.path()).error() == safetensors_errc::invalid_header_json);
    }
}

TEST_CASE("safetensors validates tensor sizes and contiguous data ranges")
{
    SUBCASE("shape byte count")
    {
        const nlohmann::json header {
            { "weight",
              {
                  { "dtype", "BF16" },
                  { "shape", { 2, 2 } },
                  { "data_offsets", { 0, 6 } },
              } },
        };
        temporary_file file("chibillm_safetensors_bad_size.safetensors", header,
                            std::vector<std::byte>(6));
        CHECK(safetensors_file::open(file.path()).error()
              == safetensors_errc::invalid_tensor_metadata);
    }

    SUBCASE("gap between tensors")
    {
        const nlohmann::json header {
            { "first",
              {
                  { "dtype", "BF16" },
                  { "shape", { 1 } },
                  { "data_offsets", { 0, 2 } },
              } },
            { "second",
              {
                  { "dtype", "BF16" },
                  { "shape", { 1 } },
                  { "data_offsets", { 4, 6 } },
              } },
        };
        temporary_file file("chibillm_safetensors_gap.safetensors", header,
                            std::vector<std::byte>(6));
        CHECK(safetensors_file::open(file.path()).error() == safetensors_errc::invalid_data_layout);
    }
}
