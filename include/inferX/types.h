#ifndef INFERX_TYPES_H
#define INFERX_TYPES_H

#include <string>

namespace inferX {

enum class DeviceType { CPU, CUDA };
enum class DType { FP16, FP32, FP64, BF16 };

struct Device {
    DeviceType type;
    int index = 0;
};

struct ModelFile {
    std::string filename;
    std::string download_url;

    ModelFile(const std::string& filename_, const std::string& download_url_)
        : filename(filename_), download_url(download_url_) {}
};

}  // namespace inferX

#endif