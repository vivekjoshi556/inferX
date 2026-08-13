#include "cpu_kernels.h"

namespace inferX {

/**
 * Naive matmul implementation:
 *
 */
Tensor cpu_matmul(const Tensor& A, const Tensor& B) { return A; }

}  // namespace inferX