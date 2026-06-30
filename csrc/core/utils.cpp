#include <cmath>
#include "utils.h"
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>

namespace inferX {

Device get_device(const std::string& device) {
    int index = 0;
    if (device.find("cuda") != std::string::npos) {
        if (device.find(":") != std::string::npos) {
            try {
                index = stoi(device.substr(device.find(":") + 1));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid Device Index in given value: " << device
                          << ". Defaulting to 0." << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Device Index out-of-range in given value: "
                          << device << ". Defaulting to 0." << std::endl;
            }
        }
        return Device({.type = DeviceType::CUDA, .index = index});
    }

    // Defaults to CPU
    return Device({.type = DeviceType::CPU, .index = index});
}

size_t get_memory_available(const Device& device) {
    switch (device.type) {
        case DeviceType::CPU:
            return cpu_available_memory(device);
        case DeviceType::CUDA:
            return cuda_available_memory(device);
        default:
            throw std::runtime_error(
                "[ERROR]: Queries mem for unsupported DeviceType");
    }
}

size_t cpu_available_memory(const Device& device) {
    // Assuming a linux based system
    std::ifstream meminfoFile("/proc/meminfo");
    std::string lineHeader;
    size_t memoryValue = 0;

    if (!meminfoFile.is_open()) {
        std::cerr << "Failed to open /proc/meminfo\n";
        return 1;
    }

    while (meminfoFile >> lineHeader) {
        if (lineHeader == "MemAvailable:") {
            if (meminfoFile >> memoryValue) {
                return memoryValue * 1024;
            }
            break;
        }
        meminfoFile.ignore(256, '\n');
    }

    return memoryValue;
}

size_t cuda_available_memory(const Device& device) {
    std::cout << "Checking device: cuda:" << device.index << " for space."
              << std::endl;
    return 0;
}

std::string human_readable_memory(const uint64_t& bytes) {
    constexpr std::string_view units[] = {"B", "KiB", "MiB", "GiB", "TiB"};

    size_t idx = 0;

    double value = static_cast<double>(bytes);
    while (value > 1024.0 && idx + 1 < std::size(units)) {
        value = ceil(value / 1024);
        idx += 1;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value << ' ' << units[idx];

    return oss.str();
}

}  // namespace inferX