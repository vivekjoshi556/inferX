#ifndef BASE_MODEL_H
#define BASE_MODEL_H

#include <unordered_map>
#include "core/layers/baseLayer.h"

namespace inferX {

class BaseModel {
    std::unordered_map<std::string, std::shared_ptr<BaseLayer>> layers_;

   public:
    BaseModel(
        std::unordered_map<std::string, std::shared_ptr<BaseLayer>>& layers)
        : layers_(layers) {}

    void add_layer(const std::string& name, std::shared_ptr<BaseLayer> layer) {
        this->layers_[name] = std::move(layer);
    }

    virtual Tensor forward(const Tensor&) const = 0;
};

}  // namespace inferX

#endif