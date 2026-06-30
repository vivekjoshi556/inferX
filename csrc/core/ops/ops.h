#ifndef INFERX_OPS_H
#define INFERX_OPS_H

#include "core/tensor.h"

namespace inferX {

namespace ops {

Tensor add(const Tensor&, const Tensor&);
Tensor matmul(const Tensor&, const Tensor&);

}  // namespace ops

}  // namespace inferX

#endif