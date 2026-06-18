#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <iostream>

namespace inferX {
void load_model(std::string model_id, std::string revision = "main",
                std::string auth_token = "");
}

#endif