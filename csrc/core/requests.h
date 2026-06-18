#ifndef REQUESTS_H
#define REQUESTS_H

#include <fstream>
#include <iostream>
#include <curl/curl.h>
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
    std::string _get_filename(std::string& url);
    void _attach_multi_handle_req(std::string, unsigned int, int&, fs::path&);
    static size_t _write_callback(char*, size_t, size_t, void*);
    static size_t _file_write_callback(char*, size_t, size_t, void*);

   public:
    Requests();

    json request_api(std::string url, std::string auth_token = "");
    void request_file(std::string url, std::string op_file);
    void request_multiple_files(std::vector<std::string>& urls, fs::path&);

    ~Requests();
};

}  // namespace inferX

#endif
