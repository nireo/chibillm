#pragma once

#include "error.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "result.h"

namespace chibillm {

enum class safetensors_dtype : std::uint8_t {
    boolean,
    u8,
    i8,
    u16,
    i16,
    f16,
    bf16,
    u32,
    i32,
    f32,
    u64,
    i64,
    f64,
    f8_e4m3,
    f8_e5m2,
};

struct safetensor_info {
    safetensors_dtype type;
    std::vector<std::size_t> shape;
    std::uint64_t data_offset;
    std::uint64_t byte_count;
};

enum class safetensors_errc : std::uint8_t {
    file_open_failed,
    file_read_failed,
    invalid_header_size,
    invalid_header_json,
    invalid_tensor_metadata,
    unsupported_dtype,
    tensor_size_overflow,
    invalid_data_layout,
    tensor_not_found,
    destination_size_mismatch,
    ambiguous_checkpoint,
    invalid_checkpoint_index,
};

[[nodiscard]] inline std::string_view
error_name(safetensors_errc code) noexcept
{
    static constexpr std::array names {
        "safetensors.file_open_failed",        "safetensors.file_read_failed",
        "safetensors.invalid_header_size",     "safetensors.invalid_header_json",
        "safetensors.invalid_tensor_metadata", "safetensors.unsupported_dtype",
        "safetensors.tensor_size_overflow",    "safetensors.invalid_data_layout",
        "safetensors.tensor_not_found",        "safetensors.destination_size_mismatch",
        "safetensors.ambiguous_checkpoint",    "safetensors.invalid_checkpoint_index",
    };
    const auto index = static_cast<std::size_t>(code);
    return index < names.size() ? names[index] : "safetensors.unknown_error";
}

class safetensors_file {
public:
    [[nodiscard]] static result<safetensors_file, safetensors_errc>
    open(const std::filesystem::path& path);

    // Prefer model.safetensors, then shards named by model.safetensors.index.json,
    // then the sole regular *.safetensors file directly in the directory.
    // Missing directories/files and filesystem errors return file_open_failed;
    // multiple fallback candidates return ambiguous_checkpoint. Preserve open() errors.
    [[nodiscard]] static result<safetensors_file, safetensors_errc>
    open_model(const std::filesystem::path& directory);

    [[nodiscard]] const safetensor_info* find(std::string_view name) const;
    [[nodiscard]] std::size_t tensor_count() const noexcept;

    [[nodiscard]] result<void, safetensors_errc> read(std::string_view name,
                                                      std::span<std::byte> destination) const;

private:
    struct source_file {
        std::filesystem::path path;
        std::uint64_t data_start;
    };

    struct tensor_entry {
        safetensor_info info;
        std::size_t file_index = 0;
    };

    safetensors_file() = default;
    safetensors_file(std::filesystem::path path,
                     std::uint64_t data_start,
                     std::unordered_map<std::string, tensor_entry> tensors);

    std::vector<source_file> files_;
    std::unordered_map<std::string, tensor_entry> tensors_;
};

} // namespace chibillm
