#ifndef ENGINE_H
#define ENGINE_H

#include <iostream>
#include "core/models/BaseModel.h"

namespace inferX {

class LLMEngine {
    std::shared_ptr<BaseModel> model_ = nullptr;

   public:
    LLMEngine(const std::string&, const std::string&, const std::string&,
              const std::string&);

    std::string generate(const std::string&);
};

}  // namespace inferX

#endif