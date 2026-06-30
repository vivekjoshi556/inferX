#ifndef LLAMA_FOR_CAUSAL_LM_H
#define LLAMA_FOR_CAUSAL_LM_H

#include "BaseModel.h"
#include "inferX/types.h"

namespace inferX {

class LlamaForCausalLM : private BaseModel {
   public:
    LlamaForCausalLM(
        const std::unordered_map<std::string, std::shared_ptr<BaseLayer>>&);
    Tensor forward(const Tensor&) const override;
};

}  // namespace inferX

#endif  // LLAMA_FOR_CAUSAL_LM_H
