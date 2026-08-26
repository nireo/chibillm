#include "model_format/safetensors.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace chibillm {
namespace {

using json = nlohmann::json;

constexpr std::uint64_t max_header_size = 100ULL * 1024ULL * 1024ULL;

std::uint64_t
decode_little_endian_u64(const std::array<std::byte, sizeof(std::uint64_t)>& bytes)
{
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[index]))
            << (index * 8);
    }
    return value;
}

result<std::uint64_t, safetensors_errc>
json_u64(const json& value)
{
    if (value.is_number_unsigned()) {
        return value.get<std::uint64_t>();
    }
    if (value.is_number_integer()) {
        const auto signed_value = value.get<std::int64_t>();
        if (signed_value >= 0) {
            return static_cast<std::uint64_t>(signed_value);
        }
    }
    return fail(safetensors_errc::invalid_tensor_metadata);
}

result<safetensors_dtype, safetensors_errc>
parse_dtype(std::string_view name)
{
    if (name == "BOOL") {
        return safetensors_dtype::boolean;
    }
    if (name == "U8") {
        return safetensors_dtype::u8;
    }
    if (name == "I8") {
        return safetensors_dtype::i8;
    }
    if (name == "U16") {
        return safetensors_dtype::u16;
    }
    if (name == "I16") {
        return safetensors_dtype::i16;
    }
    if (name == "F16") {
        return safetensors_dtype::f16;
    }
    if (name == "BF16") {
        return safetensors_dtype::bf16;
    }
    if (name == "U32") {
        return safetensors_dtype::u32;
    }
    if (name == "I32") {
        return safetensors_dtype::i32;
    }
    if (name == "F32") {
        return safetensors_dtype::f32;
    }
    if (name == "U64") {
        return safetensors_dtype::u64;
    }
    if (name == "I64") {
        return safetensors_dtype::i64;
    }
    if (name == "F64") {
        return safetensors_dtype::f64;
    }
    if (name == "F8_E4M3") {
        return safetensors_dtype::f8_e4m3;
    }
    if (name == "F8_E5M2") {
        return safetensors_dtype::f8_e5m2;
    }
    return fail(safetensors_errc::unsupported_dtype);
}

std::uint64_t
element_size(safetensors_dtype type)
{
    switch (type) {
    case safetensors_dtype::boolean:
    case safetensors_dtype::u8:
    case safetensors_dtype::i8:
    case safetensors_dtype::f8_e4m3:
    case safetensors_dtype::f8_e5m2:
        return 1;
    case safetensors_dtype::u16:
    case safetensors_dtype::i16:
    case safetensors_dtype::f16:
    case safetensors_dtype::bf16:
        return 2;
    case safetensors_dtype::u32:
    case safetensors_dtype::i32:
    case safetensors_dtype::f32:
        return 4;
    case safetensors_dtype::u64:
    case safetensors_dtype::i64:
    case safetensors_dtype::f64:
        return 8;
    }
}

result<std::vector<std::size_t>, safetensors_errc>
parse_shape(const json& value)
{
    if (!value.is_array()) {
        return fail(safetensors_errc::invalid_tensor_metadata);
    }

    std::vector<std::size_t> shape;
    shape.reserve(value.size());
    for (const auto& dimension_value : value) {
        auto dimension = json_u64(dimension_value);
        if (!dimension || *dimension > std::numeric_limits<std::size_t>::max()) {
            return fail(safetensors_errc::invalid_tensor_metadata);
        }
        shape.push_back(static_cast<std::size_t>(*dimension));
    }
    return shape;
}

result<std::uint64_t, safetensors_errc>
tensor_byte_count(const std::vector<std::size_t>& shape, safetensors_dtype type)
{
    std::uint64_t elements = 1;
    for (const auto dimension : shape) {
        if (dimension != 0 && elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
            return fail(safetensors_errc::tensor_size_overflow);
        }
        elements *= dimension;
    }

    const auto bytes_per_element = element_size(type);
    if (elements > std::numeric_limits<std::uint64_t>::max() / bytes_per_element) {
        return fail(safetensors_errc::tensor_size_overflow);
    }
    return elements * bytes_per_element;
}

result<safetensor_info, safetensors_errc>
parse_tensor_info(const json& value)
{
    if (!value.is_object()) {
        return fail(safetensors_errc::invalid_tensor_metadata);
    }

    const auto dtype_value = value.find("dtype");
    const auto shape_value = value.find("shape");
    const auto offsets_value = value.find("data_offsets");
    if (dtype_value == value.end()
        || shape_value == value.end()
        || offsets_value == value.end()
        || !dtype_value->is_string()
        || !offsets_value->is_array()
        || offsets_value->size() != 2) {
        return fail(safetensors_errc::invalid_tensor_metadata);
    }

    auto type = parse_dtype(dtype_value->get_ref<const std::string&>());
    auto shape = parse_shape(*shape_value);
    auto begin = json_u64((*offsets_value)[0]);
    auto end = json_u64((*offsets_value)[1]);
    if (!type || !shape || !begin || !end || *end < *begin) {
        return fail(!type ? type.error() : safetensors_errc::invalid_tensor_metadata);
    }

    auto expected_bytes = tensor_byte_count(*shape, *type);
    if (!expected_bytes) {
        return fail(expected_bytes.error());
    }
    if (*end - *begin != *expected_bytes) {
        return fail(safetensors_errc::invalid_tensor_metadata);
    }

    return safetensor_info {
        .type = *type,
        .shape = std::move(*shape),
        .data_offset = *begin,
        .byte_count = *expected_bytes,
    };
}

} // namespace

result<safetensors_file, safetensors_errc>
safetensors_file::open(const std::filesystem::path& path)
{
    std::error_code file_size_error;
    const auto file_size = std::filesystem::file_size(path, file_size_error);
    if (file_size_error) {
        return fail(safetensors_errc::file_open_failed);
    }
    if (file_size < sizeof(std::uint64_t)) {
        return fail(safetensors_errc::invalid_header_size);
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return fail(safetensors_errc::file_open_failed);
    }

    std::array<std::byte, sizeof(std::uint64_t)> header_size_bytes {};
    input.read(reinterpret_cast<char*>(header_size_bytes.data()), header_size_bytes.size());
    if (!input) {
        return fail(safetensors_errc::file_read_failed);
    }

    const auto header_size = decode_little_endian_u64(header_size_bytes);
    if (header_size == 0
        || header_size > max_header_size
        || header_size > file_size - sizeof(std::uint64_t)
        || header_size > std::numeric_limits<std::size_t>::max()) {
        return fail(safetensors_errc::invalid_header_size);
    }

    std::string header(static_cast<std::size_t>(header_size), '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!input) {
        return fail(safetensors_errc::file_read_failed);
    }

    const auto parsed = json::parse(header, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        return fail(safetensors_errc::invalid_header_json);
    }

    const auto data_start = sizeof(std::uint64_t) + header_size;
    const auto data_size = file_size - data_start;
    std::unordered_map<std::string, safetensor_info> tensors;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    tensors.reserve(parsed.size());
    ranges.reserve(parsed.size());

    for (const auto& [name, value] : parsed.items()) {
        if (name == "__metadata__") {
            continue;
        }

        auto info = parse_tensor_info(value);
        if (!info) {
            return fail(info.error());
        }
        if (info->data_offset > data_size || info->byte_count > data_size - info->data_offset) {
            return fail(safetensors_errc::invalid_data_layout);
        }

        ranges.emplace_back(info->data_offset, info->data_offset + info->byte_count);
        tensors.emplace(name, std::move(*info));
    }

    std::sort(ranges.begin(), ranges.end());
    std::uint64_t expected_offset = 0;
    for (const auto& [begin, end] : ranges) {
        if (begin != expected_offset) {
            return fail(safetensors_errc::invalid_data_layout);
        }
        expected_offset = end;
    }
    if (expected_offset != data_size) {
        return fail(safetensors_errc::invalid_data_layout);
    }

    return safetensors_file(path, data_start, std::move(tensors));
}

safetensors_file::safetensors_file(std::filesystem::path path,
                                   std::uint64_t data_start,
                                   std::unordered_map<std::string, safetensor_info> tensors)
    : path_(std::move(path))
    , data_start_(data_start)
    , tensors_(std::move(tensors))
{}

const safetensor_info*
safetensors_file::find(std::string_view name) const
{
    const auto found = tensors_.find(std::string(name));
    return found == tensors_.end() ? nullptr : &found->second;
}

std::size_t
safetensors_file::tensor_count() const noexcept
{
    return tensors_.size();
}

result<void, safetensors_errc>
safetensors_file::read(std::string_view name, std::span<std::byte> destination) const
{
    const auto* tensor = find(name);
    if (tensor == nullptr) {
        return fail(safetensors_errc::tensor_not_found);
    }
    if (tensor->byte_count != destination.size()) {
        return fail(safetensors_errc::destination_size_mismatch);
    }
    if (destination.empty()) {
        return {};
    }

    const auto absolute_offset = data_start_ + tensor->data_offset;
    if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())
        || destination.size()
            > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        return fail(safetensors_errc::file_read_failed);
    }

    std::ifstream input(path_, std::ios::binary);
    if (!input) {
        return fail(safetensors_errc::file_open_failed);
    }
    input.seekg(static_cast<std::streamoff>(absolute_offset));
    input.read(reinterpret_cast<char*>(destination.data()),
               static_cast<std::streamsize>(destination.size()));
    if (!input) {
        return fail(safetensors_errc::file_read_failed);
    }
    return {};
}

} // namespace chibillm
