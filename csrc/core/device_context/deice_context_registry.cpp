#include "device_context_registry.h"

namespace inferX {

std::unique_ptr<DeviceContext> DeviceContextRegistry::create_device_context(
    const Device& device) {
    switch (device.type) {
        case DeviceType::CPU:
            return std::make_unique<CPUContext>();

#ifdef INFERX_USE_CUDA
        case DeviceType::CUDA:
            return std::make_unique<CUDAContext>(device.index);
#endif

        default:
            throw std::runtime_error("Unsupported device");
    }
}

DeviceContextRegistry& DeviceContextRegistry::instance() {
    static DeviceContextRegistry instance;
    return instance;
}

DeviceContext& DeviceContextRegistry::get(const Device& device) {
    auto it = device_contexts_.find(device);

    if (it == device_contexts_.end()) {
        it = device_contexts_.emplace(device, create_device_context(device))
                 .first;
    }

    return *it->second;
}

}  // namespace inferX