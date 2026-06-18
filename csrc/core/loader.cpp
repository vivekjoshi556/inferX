#include <cstdlib>
#include "loader.h"
#include "requests.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

fs::path inferX::get_download_dir(std::string& model_id) {
    fs::path output_dir;

    const char* env_dir_path = std::getenv("INFERX_CACHE");
    if (env_dir_path != nullptr) {
        output_dir = fs::path(env_dir_path) / model_id;
    } else {
        const char* home_dir_path = std::getenv("HOME");
        output_dir =
            fs::current_path() / ".cache" / "inferX" / "hub" / model_id;
        if (home_dir_path != nullptr) {
            const char* home_env = std::getenv("HOME");
            output_dir =
                fs::path(home_env) / ".cache" / "inferX" / "hub" / model_id;
        }
    }

    return output_dir;
}

std::vector<std::string> inferX::hf_model_ls(std::string& model_id,
                                             std::string& auth_token) {
    std::string filename;
    std::vector<std::string> files;
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
        files.push_back("https://huggingface.co/" + model_id +
                        "/resolve/main/" + filename);
    }
    return files;
}

void inferX::load_model(std::string model_id, std::string revision,
                        std::string auth_token) {
    Requests download = Requests();
    std::vector<std::string> files = hf_model_ls(model_id, auth_token);

    fs::path output_dir = get_download_dir(model_id);
    std::cout << "Model Download dir:" << output_dir << std::endl;
    if (!fs::exists(output_dir) || !fs::is_directory(output_dir)) {
        fs::create_directories(output_dir);
    }

    // Downloading Model Files
    download.request_multiple_files(files, output_dir);
}