#ifndef CPU_CONTEXT_H
#define CPU_CONTEXT_H

#include "core/utils.h"
#include "device_context.h"

namespace inferX {

class CPUContext : public DeviceContext {
   public:
    std::shared_ptr<void> allocate(size_t bytes) override;

    size_t cpu_available_memory();

    void copy_from_host(const void*, const void*, const size_t,
                        const size_t offset = 0) override;
    void copy_to_host(const void*, const void*, const size_t,
                      const size_t offset = 0) override;
};

}  // namespace inferX

#endif
