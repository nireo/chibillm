#pragma once

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
};

class safetensors_file {
public:
    [[nodiscard]] static result<safetensors_file, safetensors_errc>
    open(const std::filesystem::path& path);

    [[nodiscard]] const safetensor_info* find(std::string_view name) const;
    [[nodiscard]] std::size_t tensor_count() const noexcept;

    [[nodiscard]] result<void, safetensors_errc> read(std::string_view name,
                                                      std::span<std::byte> destination) const;

private:
    safetensors_file(std::filesystem::path path,
                     std::uint64_t data_start,
                     std::unordered_map<std::string, safetensor_info> tensors);

    std::filesystem::path path_;
    std::uint64_t data_start_;
    std::unordered_map<std::string, safetensor_info> tensors_;
};

} // namespace chibillm
