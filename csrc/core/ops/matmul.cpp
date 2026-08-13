#include "ops.h"
#include "core/dispatcher.h"

namespace inferX {

namespace ops {

/**
 * Matrix Multiplication:
 *
 * - If both tensors are 1-dimensional, the dot product (scalar) is returned.
 * - If both arguments are 2-dimensional, the matrix-matrix product is returned.
 * - If the first argument is 1-dimensional and the second argument is
 * 2-dimensional, a 1 is prepended to its dimension for the purpose of the
 * matrix multiply. After the matrix multiply, the prepended dimension is
 * removed.
 * - If the first argument is 2-dimensional and the second argument is
 * 1-dimensional, the matrix-vector product is returned.
 * - If both arguments are at least 1-dimensional and at least one argument is
 * N-dimensional (where N > 2), then a batched matrix multiply is returned. If
 * the first argument is 1-dimensional, a 1 is prepended to its dimension for
 * the purpose of the batched matrix multiply and removed after. If the second
 * argument is 1-dimensional, a 1 is appended to its dimension for the purpose
 * of the batched matrix multiply and removed after.
 * - The first N-2 dimensions of each argument, the batch dimensions, are
 * broadcast (and thus must be broadcastable). The last 2, the matrix
 * dimensions, are handled as in the matrix-matrix product.
 *
 */
Tensor matmul(const Tensor& A, const Tensor& B) {
    const auto& a_shape = A.shape();
    const auto& b_shape = B.shape();

    const size_t a_dim = a_shape.size();
    const size_t b_dim = b_shape.size();

    // Case 1: 1D @ 1D → dot product (scalar)
    if (a_dim == 1 && b_dim == 1) {
        if (a_shape[0] != b_shape[0]) {
            throw std::runtime_error(
                "[ERROR]: 1D dot product requires equal lengths, got (" +
                std::to_string(a_shape[0]) + ") @ (" +
                std::to_string(b_shape[0]) + ")");
        }
        return dispatcher::matmul(A, B);
    }

    // Case 2: 2D @ 2D → matrix-matrix
    if (a_dim == 2 && b_dim == 2) {
        if (a_shape[1] != b_shape[0]) {
            throw std::runtime_error(
                "[ERROR]: Matrix-matrix multiply shape mismatch, got (" +
                std::to_string(a_shape[0]) + ", " + std::to_string(a_shape[1]) +
                ") @ (" + std::to_string(b_shape[0]) + ", " +
                std::to_string(b_shape[1]) + ")");
        }
        return dispatcher::matmul(A, B);
    }

    // Case 3: 1D @ 2D → (1, K) @ (K, N) → (N)
    if (a_dim == 1 && b_dim == 2) {
        if (a_shape[0] != b_shape[0]) {
            throw std::runtime_error(
                "[ERROR]: 1D @ 2D matmul requires A[0] == B[0], got (" +
                std::to_string(a_shape[0]) + ") @ (" +
                std::to_string(b_shape[0]) + ", " + std::to_string(b_shape[1]) +
                ")");
        }
        return dispatcher::matmul(A, B);
    }

    // Case 4: 2D @ 1D → (M, K) @ (K) → (M)
    if (a_dim == 2 && b_dim == 1) {
        if (a_shape[1] != b_shape[0]) {
            throw std::runtime_error(
                "[ERROR]: 2D @ 1D matmul requires A[1] == B[0], got (" +
                std::to_string(a_shape[0]) + ", " + std::to_string(a_shape[1]) +
                ") @ (" + std::to_string(b_shape[0]) + ")");
        }
        return dispatcher::matmul(A, B);
    }

    // Case 5: Batched matmul (N-D where N > 2)
    if (a_dim >= 1 && b_dim >= 1 && (a_dim > 2 || b_dim > 2)) {
        // Effective matrix dims
        const size_t a_k = (a_dim == 1) ? a_shape[0] : a_shape[a_dim - 1];
        const size_t b_k = (b_dim == 1) ? b_shape[0] : b_shape[b_dim - 2];

        if (a_k != b_k) {
            throw std::runtime_error(
                "[ERROR]: Batched matmul inner dimensions mismatch, got K=" +
                std::to_string(a_k) + " and K=" + std::to_string(b_k));
        }

        return dispatcher::matmul(A, B);
    }

    throw std::runtime_error("[ERROR]: Unsupported tensor ranks for matmul");
}

}  // namespace ops

}  // namespace inferX