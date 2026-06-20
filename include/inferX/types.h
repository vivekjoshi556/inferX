#ifndef INFERX_TYPES_H
#define INFERX_TYPES_H

#include <string>

namespace inferX {

struct ModelFile {
    std::string filename;
    std::string download_url;

    ModelFile(const std::string& filename_, const std::string& download_url_)
        : filename(filename_), download_url(download_url_) {}
};

}  // namespace inferX

#endif