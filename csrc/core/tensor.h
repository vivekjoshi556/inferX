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
    std::shared_ptr<void> data_;
    DType dtype_;
    std::string name_;
    std::vector<uint64_t> shape_;
    Device device_;

    std::shared_ptr<void> _get_memory(const uint64_t&, const Device&);

   public:
    Tensor(const SafeTensorEntry&, const Device&);
    Tensor(const std::vector<uint64_t>&, const Device&, const DType&);

    // Getters
    const DType& dtype() const { return dtype_; }
    const std::string& name() const { return name_; }
    const std::vector<uint64_t>& shape() const { return shape_; }
    Device device() const { return device_; }

    template <typename T>
    T* data() {
        return static_cast<T*>(data_.get());
    }

    template <typename T>
    const T* data() const {
        return static_cast<const T*>(data_.get());
    }
};

class SafeTensorFile {
    std::unordered_map<std::string, SafeTensorEntry> tensor_list_;

    DType _get_dtype(const std::string&);

   public:
    explicit SafeTensorFile(const fs::path&);

    const std::unordered_map<std::string, SafeTensorEntry> get_tensor_list()
        const {
        return this->tensor_list_;
    }

    const SafeTensorEntry& get_entry(const std::string& name) const {
        return this->tensor_list_.at(name);
    };

    std::unordered_map<std::string, Tensor> load_tensors(const Device&);
};

}  // namespace inferX

#endif