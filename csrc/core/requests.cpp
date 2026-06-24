#include <filesystem>
#include "requests.h"
#include <nlohmann/json.hpp>

inferX::Requests::Requests() {
    CURLcode result = curl_global_init(CURL_GLOBAL_ALL);

    if (result != CURLE_OK) {
        std::cerr << "[ERROR]: cURL global init failed" << std::endl;
        return;
    }

    this->_init_curl();
    this->_init_multi_curl();
}

size_t inferX::Requests::_write_callback(char* ptr, size_t size, size_t nmemb,
                                         void* userdata) {
    auto* data = static_cast<std::string*>(userdata);

    data->append(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}

size_t inferX::Requests::_file_write_callback(char* ptr, size_t size,
                                              size_t nmemb, void* userdata) {
    auto* file = static_cast<std::ofstream*>(userdata);

    file->write(ptr, static_cast<std::streamsize>(size * nmemb));

    return size * nmemb;
}

void inferX::Requests::_init_curl() {
    if (this->handle) return;

    this->handle = curl_easy_init();
    if (!this->handle) {
        std::cerr << "Failed to initialize cURL" << std::endl;
    }
}

void inferX::Requests::_init_multi_curl() {
    if (this->multi_handle) return;

    this->multi_handle = curl_multi_init();
    if (!this->multi_handle) {
        std::cerr << "Failed to initialize multi-cURL" << std::endl;
    }
}

json inferX::Requests::request_api(const std::string& url,
                                   const std::string& auth_token) {
    json result;

    if (!this->handle) {
        this->_init_curl();

        if (!this->handle) return result;
    }

    CURLcode response;
    std::string readBuffer;

    struct curl_slist* headers = nullptr;

    if (auth_token.size() > 0)
        headers = curl_slist_append(
            headers, ("Authorization: Bearer " + auth_token).c_str());

    curl_easy_setopt(this->handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(this->handle, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(this->handle, CURLOPT_WRITEDATA, &readBuffer);
    curl_easy_setopt(this->handle, CURLOPT_WRITEFUNCTION,
                     this->_write_callback);

    response = curl_easy_perform(this->handle);

    if (response != CURLE_OK) {
        std::cerr << "Request Failed: " << curl_easy_strerror(response)
                  << std::endl;
        return result;
    }

    result = json::parse(readBuffer, nullptr, false);
    if (result.is_discarded()) {
        std::cerr << "[ERROR]: Failed to parse API Response" << std::endl;
        return result;
    }

    return result;
}

bool inferX::Requests::_attach_multi_handle_req(const inferX::ModelFile& file,
                                                const fs::path& output_dir) {
    // First check if the file already exists
    std::string op_filepath = output_dir / file.filename;
    if (fs::exists(op_filepath) && fs::is_regular_file(op_filepath)) {
        return false;
    }

    CURL* local_curl = curl_easy_init();
    if (!local_curl) {
        std::cerr << "[ERROR]: Failed to initialize cURL for filename: "
                  << file.filename << std::endl;
        return false;
    }

    auto* ctx = new DownloadContext();
    ctx->url = file.download_url;

    ctx->file.open(op_filepath, std::ios::binary);
    if (!ctx->file.is_open()) {
        std::cerr << "[ERROR]: Failed to open file: " << op_filepath
                  << std::endl;
        curl_easy_cleanup(local_curl);
        delete ctx;
        return false;
    }

    curl_easy_setopt(local_curl, CURLOPT_URL, ctx->url.c_str());
    curl_easy_setopt(local_curl, CURLOPT_WRITEDATA, &ctx->file);
    curl_easy_setopt(local_curl, CURLOPT_WRITEFUNCTION,
                     this->_file_write_callback);

    curl_easy_setopt(local_curl, CURLOPT_TIMEOUT, 0);
    curl_easy_setopt(local_curl, CURLOPT_PRIVATE, ctx);

    curl_easy_setopt(local_curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(local_curl, CURLOPT_MAXREDIRS, 10L);

    // curl_easy_setopt(local_curl, CURLoption option, parameter);
    curl_easy_setopt(local_curl, CURLOPT_NOPROGRESS, 0L);

    curl_multi_add_handle(this->multi_handle, local_curl);
    return true;
}

void inferX::Requests::request_multiple_files(
    const std::vector<inferX::ModelFile>& files, const fs::path& output_dir) {
    json result;

    if (!this->multi_handle) {
        this->_init_multi_curl();

        if (!this->multi_handle) {
            std::cerr << "[ERROR]: File Download failed. Could not initialize "
                         "multi-handler.";
            return;
        }
    }

    CURLMsg* msg;
    fs::path op_filepath;
    unsigned int transfers = 0;
    int msgs_left = 0;
    int left = 0;

    curl_multi_setopt(this->multi_handle, CURLMOPT_MAXCONNECTS,
                      (long)MAX_PARALLEL);

    long transfers_it_limit =
        std::min((long)MAX_PARALLEL, (long)(files.size()));

    // Adding transfers to multi-curl.
    for (transfers = 0; transfers < files.size() && left < transfers_it_limit;
         transfers++) {
        op_filepath = output_dir / files[transfers].filename;
        if (this->_attach_multi_handle_req(files[transfers], output_dir)) {
            left += 1;
        }
    }

    while (left) {
        int still_alive = 1;
        curl_multi_perform(this->multi_handle, &still_alive);

        while ((msg = curl_multi_info_read(this->multi_handle, &msgs_left)) !=
               NULL) {
            if (msg->msg == CURLMSG_DONE) {
                CURL* curl = msg->easy_handle;
                DownloadContext* ctx = nullptr;

                curl_easy_getinfo(curl, CURLINFO_PRIVATE, &ctx);
                if (ctx) {
                    ctx->file.close();
                    delete ctx;
                }
                curl_multi_remove_handle(this->multi_handle, curl);
                curl_easy_cleanup(curl);
                left--;

                if (transfers < files.size()) {
                    op_filepath = output_dir / files[transfers].filename;
                    if (this->_attach_multi_handle_req(files[transfers],
                                                       output_dir)) {
                        left += 1;
                    }
                    transfers += 1;
                }
            }
        }
        if (left) curl_multi_wait(this->multi_handle, NULL, 0, 1000, NULL);
    }

    curl_multi_cleanup(this->multi_handle);
    this->multi_handle = nullptr;
}

inferX::Requests::~Requests() {
    curl_global_cleanup();
    if (this->multi_handle) {
        curl_multi_cleanup(this->multi_handle);
        this->multi_handle = nullptr;
    }
    if (this->handle) {
        curl_easy_cleanup(this->handle);
        this->handle = nullptr;
    }
}
