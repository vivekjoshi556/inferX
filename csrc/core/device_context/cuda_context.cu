#include "core/utils.h"
#include "inferX/types.h"
#include "cuda_context.h"

namespace inferX {

std::shared_ptr<void> CUDAContext::allocate(size_t bytes) {
    void* ptr = nullptr;

    cudaError_t err = cudaMalloc(&ptr, bytes);

    if (err != cudaSuccess) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;

        cudaMemGetInfo(&free_bytes, &total_bytes);

        throw std::runtime_error(
            "CUDA out of memory: tried to allocate " +
            human_readable_memory(bytes) +
            ", free: " + human_readable_memory(free_bytes) +
            ", total: " + human_readable_memory(total_bytes) +
            ". CUDA error: " + cudaGetErrorString(err));
    }

    return std::shared_ptr<void>(ptr, [](void* p) { cudaFree(p); });
}

void copy_from_host(const void* src, const void* dest,
                            const size_t n_bytes, const size_t offset = 0) {
    //
}

void copy_to_host(const void* src, const void* dest,
                          const size_t n_bytes, const size_t offset = 0) {
    //
}

}  // namespace inferX