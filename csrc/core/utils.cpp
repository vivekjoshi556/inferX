#include <cmath>
#include "utils.h"
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