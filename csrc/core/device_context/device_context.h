#ifndef DEVICE_CONTEXT_H
#define DEVICE_CONTEXT_H

#include <memory>
#include <stddef.h>
#include "inferX/types.h"

namespace inferX {

class DeviceContext {
   protected:
    size_t allocated_bytes = 0;
    Device device_;

   public:
    virtual ~DeviceContext() = default;

    virtual std::shared_ptr<void> allocate(size_t bytes) = 0;

    virtual void copy_from_host(const void* src, const void* dest,
                                const size_t n_bytes,
                                const size_t offset = 0) = 0;
    virtual void copy_to_host(const void* src, const void* dest,
                              const size_t n_bytes,
                              const size_t offset = 0) = 0;
};

}  // namespace inferX

#endif