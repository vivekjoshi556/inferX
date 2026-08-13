#include "core/utils.h"
#include "loader.h"
#include "engine.h"

namespace inferX {

LLMEngine::LLMEngine(const std::string& model_id, const std::string& revision,
                     const std::string& auth_token, const std::string& device) {
    load_model(model_id, revision, auth_token, device);
    // init model_ next.
};

std::string LLMEngine::generate(const std::string& ip) { return ip; }

}  // namespace inferX
