#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <vector>

#include "model_format/safetensors.h"
#include "safetensors_test_support.h"

using chibillm::safetensors_dtype;
using chibillm::safetensors_errc;
using chibillm::safetensors_file;
using safetensors_test::temporary_file;

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
