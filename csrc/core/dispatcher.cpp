#include "dispatcher.h"
#include "core/ops/ops.h"
#include "kernels/cpu/cpu_kernels.h"

namespace inferX::dispatcher {

Tensor add(const Tensor& A, const Tensor& B) {
    // check and call correct implementation of op
    if (A.device().type != B.device().type)
        throw std::runtime_error(
            "Cannot perform op::matmul on tensors of different devices");

    // Later based on device type
    return cpu_add(A, B);
}

Tensor matmul(const Tensor& A, const Tensor& B) {
    // check and call correct implementation of op
    if (A.device().type != B.device().type)
        throw std::runtime_error(
            "Cannot perform op::matmul on tensors of different devices");

    // Later based on device type
    return cpu_matmul(A, B);
}

}  // namespace inferX::dispatcher