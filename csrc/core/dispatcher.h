#ifndef DISPATCHER_H
#define DISPATCHER_H

#include "core/ops/ops.h"
#include "tensor.h"

namespace inferX {

namespace dispatcher {

Tensor add(const Tensor&, const Tensor&);
Tensor matmul(const Tensor&, const Tensor&);

}  // namespace dispatcher

}  // namespace inferX

#endif