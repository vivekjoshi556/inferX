#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "core/tensor.h"

namespace inferX {

class BaseLayer {
    std::string name;

   public:
    virtual Tensor forward(const Tensor&) const = 0;
    virtual ~BaseLayer() = default;
};

}  // namespace inferX

#endif