#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "baseLayer.h"

namespace inferX {

class LinearLayer : public BaseLayer {
    const Tensor* weights_ = nullptr;
    const Tensor* biases_ = nullptr;

   public:
    LinearLayer(const Tensor&);
    LinearLayer(const Tensor&, const Tensor&);

    Tensor forward(const Tensor&) const override;
};

}  // namespace inferX

#endif