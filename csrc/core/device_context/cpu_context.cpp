#include <fstream>
#include "cpu_context.h"

namespace inferX {

std::shared_ptr<void> CPUContext::allocate(size_t bytes) {
    void* ptr = std::malloc(bytes);
    this->allocated_bytes += bytes;

    if (!ptr) {
        size_t available_memory = cpu_available_memory();
        throw std::runtime_error(
            "[ERROR]: Device out of memory. Tried to allocate " +
            human_readable_memory(bytes) +
            ", available: " + human_readable_memory(available_memory));
    }

    return std::shared_ptr<void>(ptr, [this, bytes](void* p) {
        this->allocated_bytes -= bytes;
        std::free(p);
    });
}

void CPUContext::copy_from_host(const void* src, const void* dest,
                                const size_t n_bytes, const size_t offset) {
    std::cout << "[WARNING]: Trying to copy from device CPU to CPU"
              << std::endl;
}

void CPUContext::copy_to_host(const void* src, const void* dest,
                              const size_t n_bytes, const size_t offset) {
    std::cout << "[WARNING]: Trying to copy from device CPU to CPU"
              << std::endl;
}

size_t CPUContext::cpu_available_memory() {
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

}  // namespace inferX