#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

#include "core/tensor.h"

namespace inferX {

Tensor cpu_add(const Tensor&, const Tensor&);
Tensor cpu_matmul(const Tensor&, const Tensor&);

}  // namespace inferX

#endif