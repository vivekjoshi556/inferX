#ifndef DEVICE_CONTEXT_REGISTRY_H
#define DEVICE_CONTEXT_REGISTRY_H

#include <mutex>
#include <memory>
#include <unordered_map>

#include "device_context.h"
#include "cpu_context.h"
#include "inferX/types.h"

#ifdef INFERX_USE_CUDA
    #include "cuda_context.h"
#endif

namespace inferX {
class DeviceContextRegistry {
    std::unordered_map<Device, std::unique_ptr<DeviceContext>> device_contexts_;

    DeviceContextRegistry() = default;
    ~DeviceContextRegistry() = default;

    DeviceContextRegistry(const DeviceContextRegistry&) = delete;
    DeviceContextRegistry& operator=(const DeviceContextRegistry&) = delete;

    std::unique_ptr<DeviceContext> create_device_context(const Device&);

   public:
    static DeviceContextRegistry& instance();
    DeviceContext& get(const Device&);
};

}  // namespace inferX

#endif