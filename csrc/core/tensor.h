#ifndef INFERX_TENSOR_H
#define INFERX_TENSOR_H

#include <vector>
#include <iostream>
#include <stdint.h>
#include <filesystem>
#include <unordered_map>
#include "inferX/types.h"

namespace fs = std::filesystem;

namespace inferX {
struct SafeTensorEntry {
    std::string name;
    DType dtype;
    fs::path filepath;
    std::vector<uint64_t> shape;
    uint64_t offset_start;
    uint64_t offset_end;
};

class Tensor {
   public:
    void* data;
    DType dtype;
    std::string name;
    std::vector<uint64_t> shape;
    Device device;

    Tensor(const SafeTensorEntry&, const Device&);
    ~Tensor();
};

class SafeTensorFile {
    std::unordered_map<std::string, SafeTensorEntry> tensor_list;

    DType _get_dtype(const std::string&);

   public:
    explicit SafeTensorFile(const fs::path&);

    const std::unordered_map<std::string, SafeTensorEntry> get_tensor_list()
        const;

    const SafeTensorEntry& get_entry(const std::string&) const;

    std::unordered_map<std::string, Tensor> load_tensors(const Device&);
};

}  // namespace inferX

#endif