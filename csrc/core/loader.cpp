#include "utils.h"
#include <cstdlib>
#include "tensor.h"
#include "loader.h"
#include "requests.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace inferX {

fs::path get_download_dir(const std::string& model_id,
                          const std::string& revision) {
    fs::path output_dir;

    const char* env_dir_path = std::getenv("INFERX_CACHE");
    if (env_dir_path != nullptr) {
        output_dir = fs::path(env_dir_path) / model_id;
    } else {
        const char* home_dir_path = std::getenv("HOME");
        output_dir =
            fs::current_path() / ".cache" / "inferX" / "hub" / model_id;
        if (home_dir_path != nullptr) {
            output_dir = fs::path(home_dir_path) / ".cache" / "inferX" / "hub" /
                         model_id;
        }
    }

    output_dir /= revision;

    std::cout << "Model Dir: " << output_dir;
    if (fs::create_directories(output_dir))
        std::cout << " created successfully" << std::endl;
    else
        std::cout << " already exists" << std::endl;

    return output_dir;
}

std::vector<ModelFile> hf_model_ls(const std::string& model_id,
                                   const std::string& revision,
                                   const std::string& auth_token) {
    std::string filename;
    std::vector<ModelFile> files;
    Requests download = Requests();

    const std::string HF_BASE_API_URL = "https://huggingface.co/api/models";

    json response =
        download.request_api(HF_BASE_API_URL + "/" + model_id, auth_token);
    if (response.is_discarded()) {
        std::cerr << "[ERROR]: Failed to get Model Details";
        return files;
    }

    if (!response.contains("siblings") || !response["siblings"].is_array()) {
        std::cerr << "[ERROR]: Could not retrieve list of files for model_id: "
                  << model_id << std::endl;
        return files;
    }

    for (auto& it : response["siblings"]) {
        if (it.is_object() && it.contains("rfilename")) {
            filename = it["rfilename"].get<std::string>();
        } else if (it.is_string()) {
            filename = it.get<std::string>();
        }

        files.emplace_back(filename, "https://huggingface.co/" + model_id +
                                         "/resolve/main/" + filename);
    }
    return files;
}

std::unordered_map<std::string, Tensor> load_safetensors(
    const fs::path& model_dir, const Device& device) {
    std::unordered_map<std::string, Tensor> tensors;
    for (auto file : fs::directory_iterator(model_dir)) {
        std::string filepath = file.path().c_str();
        if (filepath.find(".safetensor") != std::string::npos) {
            SafeTensorFile tensor_file(file);

            tensors.merge(tensor_file.load_tensors(device));
        }
    }

    return tensors;
}

void download_model(const std::string& model_id, const fs::path& output_dir,
                    const std::string& revision,
                    const std::string& auth_token) {
    Requests download = Requests();
    std::vector<ModelFile> files = hf_model_ls(model_id, revision, auth_token);

    if (!fs::exists(output_dir) || !fs::is_directory(output_dir)) {
        fs::create_directories(output_dir);
    }

    // Check for sub-directories
    for (ModelFile& file : files) {
        auto idx = file.filename.rfind("/");
        if (idx != std::string::npos) {
            fs::path subdir_path = output_dir / file.filename.substr(0, idx);
            if (!fs::exists(subdir_path) || !fs::is_directory(subdir_path)) {
                fs::create_directories(subdir_path);
            }
        }
    }
    // Downloading Model Files
    download.request_multiple_files(files, output_dir, auth_token);
}

void load_model(const std::string& model_id, const std::string& revision,
                const std::string& auth_token, const std::string& device) {
    fs::path output_dir = get_download_dir(model_id);

    // download_model(model_id, output_dir, revision, auth_token);

    Device d = get_device(device);
    std::unordered_map<std::string, Tensor> tensors =
        load_safetensors(output_dir, d);
}

}  // namespace inferX