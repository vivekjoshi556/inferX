#include "linear.h"
#include "core/ops/ops.h"

namespace inferX {

LinearLayer::LinearLayer(const Tensor& weights) : weights_(&weights) {}
LinearLayer::LinearLayer(const Tensor& weights, const Tensor& biases)
    : weights_(&weights), biases_(&biases) {}

Tensor LinearLayer::forward(const Tensor& input) const {
    Tensor result = ops::matmul(*this->weights_, input);

    if (this->biases_) {
        result = ops::add(result, *this->biases_);
    }

    return result;
}

}  // namespace inferX