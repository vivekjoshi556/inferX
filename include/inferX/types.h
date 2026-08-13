#ifndef INFERX_TYPES_H
#define INFERX_TYPES_H

#include <string>
#include <iostream>
#include <functional>

namespace inferX {

enum class DeviceType { CPU, CUDA };
enum class DType { FP16, FP32, FP64, BF16 };

constexpr size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FP16:
        case DType::BF16:
            return 2;
        case DType::FP32:
            return 4;
        case DType::FP64:
            return 8;
    }

    throw std::runtime_error("Unknown DType");
}

struct Device {
    DeviceType type;
    int index = 0;

    bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }
};

struct ModelFile {
    std::string filename;
    std::string download_url;

    ModelFile(const std::string& filename_, const std::string& download_url_)
        : filename(filename_), download_url(download_url_) {}
};

}  // namespace inferX

template <>
struct std::hash<inferX::Device> {
    std::size_t operator()(const inferX::Device& d) const noexcept {
        std::size_t h1 = std::hash<inferX::DeviceType>{}(d.type);
        std::size_t h2 = std::hash<int>{}(d.index);
        return h1 ^ (h2 << 1);  // or use boost::hash_combine
    }
};

#endif