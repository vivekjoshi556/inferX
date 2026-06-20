#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <vector>
#include <iostream>
#include <filesystem>
#include "inferX/types.h"

namespace fs = std::filesystem;

namespace inferX {
std::vector<inferX::ModelFile> hf_model_ls(const std::string&,
                                           const std::string&);
fs::path get_download_dir(const std::string&);
void load_model(const std::string&, const std::string& revision = "main",
                const std::string& auth_token = "");
}  // namespace inferX

#endif