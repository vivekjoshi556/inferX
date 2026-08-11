#ifndef INFERX_TENSOR_H
#define INFERX_TENSOR_H

#include <vector>
#include <iostream>
#include <stdint.h>
#include <filesystem>
#include <unordered_map>
#include "inferX/types.h"
#include "device_context/device_context_registry.h"

namespace fs = std::filesystem;

namespace inferX {
typedef std::vector<uint64_t> TensorShape;

struct SafeTensorEntry {
    std::string name;
    DType dtype;
    fs::path filepath;
    TensorShape shape;
    uint64_t offset_start;
    uint64_t offset_end;
};

class TensorStorage {
   private:
    std::shared_ptr<void> data_;
    Device device_;
    DType dtype_;
    size_t nbytes_;
    DeviceContext& d_ctx_;

   public:
    TensorStorage(const Device& device, const size_t req_mem,
                  const DType& dtype);

    // Getters
    const DType& dtype() const { return dtype_; }
    Device device() const { return device_; }

    void* raw_data() { return data_.get(); }
    const void* raw_data() const { return data_.get(); }

    template <typename T>
    T* data() {
        return static_cast<T*>(data_.get());
    }

    template <typename T>
    const T* data() const {
        return static_cast<const T*>(data_.get());
    }

    void copy_from_host(char* src, const size_t num_bytes,
                        const size_t offset = 0) {
        this->d_ctx_.copy_from_host(src, this->data<char>() + offset, num_bytes,
                                    offset);
    }

    void copy_to_host(char* src, const size_t num_bytes,
                      const size_t offset = 0) {
        this->d_ctx_.copy_from_host(src, this->data<char>() + offset, num_bytes,
                                    offset);
    }
};

class Tensor {
    std::string name_;
    TensorShape shape_;
    std::shared_ptr<TensorStorage> storage_;

   public:
    // Tensor(const SafeTensorEntry&, const Device&);
    Tensor(const TensorShape&, const Device&, const DType&);

    // Getters
    const std::string& name() const { return name_; }
    const TensorShape& shape() const { return shape_; }

    Device device() const { return storage_->device(); }
    const DType& dtype() const { return storage_->dtype(); }

    template <typename T>
    T* data() {
        return storage_->data<T>();
    }

    template <typename T>
    const T* data() const {
        return storage_->data<T>();
    }

    TensorStorage& storage() { return *storage_; }
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

    Tensor load_tensor(const SafeTensorEntry&, const Device&);
    std::unordered_map<std::string, Tensor> load_tensors(const Device&);
};

}  // namespace inferX

#endif