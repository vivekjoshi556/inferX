#include "ops.h"
#include "core/dispatcher.h"

namespace inferX {

namespace ops {

Tensor add(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 2 || A.shape().size() != B.shape().size() ||
        A.shape()[1] != B.shape()[0]) {
        std::runtime_error(
            "[ERROR]: i/p tensor shape mismatch for add, given shapes are (" +
            std::to_string(A.shape()[0]) + ", " + std::to_string(A.shape()[1]) +
            ") @ " + std::to_string(B.shape()[0]) + ", " +
            std::to_string(B.shape()[1]) + ")");
    }

    return dispatcher::add(A, B);
}

}  // namespace ops

}  // namespace inferX