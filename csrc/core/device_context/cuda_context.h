#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H

#include "device_context.h"

namespace inferX {

class CUDAContext : public DeviceContext {
   public:
    std::shared_ptr<void> allocate(size_t bytes) override;

    void copy_from_host(void*, const void*, const size_t,
                        const size_t offset = 0) override;
    void copy_to_host(void*, const void*, const size_t,
                      const size_t offset = 0) override;
};

}  // namespace inferX

#endif