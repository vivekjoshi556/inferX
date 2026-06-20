#ifndef REQUESTS_H
#define REQUESTS_H

#include <fstream>
#include <iostream>
#include <curl/curl.h>
#include "inferX/types.h"
#include <nlohmann/json.hpp>

#define MAX_PARALLEL 5

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace inferX {
struct DownloadContext {
    std::ofstream file;
    std::string url;
};

class Requests {
    CURL* handle = nullptr;
    CURLM* multi_handle = nullptr;

    void _init_curl();
    void _init_multi_curl();
    bool _attach_multi_handle_req(const inferX::ModelFile& file,
                                  const fs::path& output_dir);
    static size_t _write_callback(char*, size_t, size_t, void*);
    static size_t _file_write_callback(char*, size_t, size_t, void*);

   public:
    Requests();

    json request_api(const std::string& url,
                     const std::string& auth_token = "");
    void request_multiple_files(const std::vector<ModelFile>&, const fs::path&);

    ~Requests();
};

}  // namespace inferX

#endif
