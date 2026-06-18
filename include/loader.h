#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace inferX {
fs::path get_download_dir(std::string& model_id);
void load_model(std::string model_id, std::string revision = "main",
                std::string auth_token = "");
}  // namespace inferX

#endif