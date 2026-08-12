#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H

#include "device_context.h"

namespace inferX {

class CUDAContext : public DeviceContext {
    int index_ = 0;

   public:
    CUDAContext(int index = 0) : index_(index) {}
    std::shared_ptr<void> allocate(size_t bytes) override;

    void copy_from_host(const void* src, const void* dest, const size_t n_bytes,
                        const size_t offset = 0) override;
    void copy_to_host(const void* src, const void* dest, const size_t n_bytes,
                      const size_t offset = 0) override;
};

}  // namespace inferX

#endif