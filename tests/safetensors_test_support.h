#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace safetensors_test {

class temporary_file {
public:
    temporary_file(std::string name,
                   const nlohmann::json& header,
                   const std::vector<std::byte>& data)
        : path_(std::filesystem::temp_directory_path() / std::move(name))
    {
        auto header_text = header.dump();
        while (header_text.size() % 8 != 0) {
            header_text.push_back(' ');
        }

        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        const auto header_size = static_cast<std::uint64_t>(header_text.size());
        std::array<std::byte, sizeof(header_size)> encoded_size {};
        for (std::size_t index = 0; index < encoded_size.size(); ++index) {
            encoded_size[index] = static_cast<std::byte>((header_size >> (index * 8)) & 0xFF);
        }
        output.write(reinterpret_cast<const char*>(encoded_size.data()), encoded_size.size());
        output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
        output.write(reinterpret_cast<const char*>(data.data()),
                     static_cast<std::streamsize>(data.size()));
    }

    temporary_file(const temporary_file&) = delete;
    temporary_file& operator=(const temporary_file&) = delete;

    ~temporary_file()
    {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path&
    path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

inline void
add_tensor(nlohmann::json& header,
           std::vector<std::byte>& data,
           std::string name,
           std::string dtype,
           std::vector<std::size_t> shape)
{
    std::size_t elements = 1;
    for (const auto dimension : shape) {
        elements *= dimension;
    }
    const auto element_bytes = dtype == "F32" ? 4U : 2U;
    const auto begin = data.size();
    data.resize(begin + elements * element_bytes);
    header[std::move(name)] = {
        { "dtype", std::move(dtype) },
        { "shape", std::move(shape) },
        { "data_offsets", { begin, data.size() } },
    };
}

} // namespace safetensors_test
