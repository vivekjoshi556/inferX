#include "utils.h"
#include "tensor.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace inferX {

DType SafeTensorFile::_get_dtype(const std::string& type) {
    if (type == "F16") return DType::FP16;
    if (type == "F32") return DType::FP32;
    if (type == "F64") return DType::FP64;
    if (type == "BF16") return DType::BF16;

    throw std::runtime_error("[ERROR]: Unsupported dtype: " + type);
}

SafeTensorFile::SafeTensorFile(const fs::path& filepath) {
    std::ifstream handle(filepath, std::ios::binary);

    if (!handle) {
        std::cerr << "[ERROR]: Could not open file: " << filepath << std::endl;
        return;
    }

    uint64_t header_size = 0;
    handle.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    std::cout << "Header size: " << header_size << std::endl;

    std::string headers(header_size, '\0');
    handle.read(&headers[0], static_cast<std::streamsize>(header_size));
    nlohmann::json headers_json = json::parse(headers);

    for (auto& el : headers_json.items()) {
        if (el.key() == std::string("__metadata__")) {
            continue;
        } else {
            // Check if all the keys exist.
            // shape, dtype, data_offsets
            if (!el.value().is_object()) {
                std::cout
                    << "[WARNING]: Expecting Safetensor Entry to be an object"
                    << std::endl;
                continue;
            }

            bool propsPresent = true;
            for (std::string prop : {"dtype", "shape", "data_offsets"}) {
                if (!el.value().contains(prop)) {
                    std::cout << "[WARNING]: Missing key=" << prop
                              << " in Safetensor Entry" << std::endl;
                    propsPresent = false;
                }
            }

            if (!propsPresent) continue;

            std::vector<uint64_t> data_offset =
                el.value()["data_offsets"].get<std::vector<uint64_t>>();

            if (data_offset.size() != 2) {
                std::cout << "[WARNING]: data_offsets must contain exactly 2 "
                             "elements"
                          << std::endl;
                continue;
            }

            this->tensor_list_[el.key()] = SafeTensorEntry(
                {.name = el.key(),
                 .dtype = _get_dtype(el.value()["dtype"].get<std::string>()),
                 .filepath = filepath,
                 .shape = el.value()["shape"].get<std::vector<uint64_t>>(),
                 .offset_start = data_offset[0],
                 .offset_end = data_offset[1]});
        }
    }

    handle.close();
}

std::unordered_map<std::string, Tensor> SafeTensorFile::load_tensors(
    const Device& device) {
    std::unordered_map<std::string, Tensor> tensors;

    for (auto& el : this->tensor_list_) {
        tensors.try_emplace(el.first, el.second, device);
    }

    return tensors;
}

std::shared_ptr<void> Tensor::_get_memory(const uint64_t& required_bytes,
                                          const Device& device) {
    uint64_t available_memory = get_memory_available(device);
    if (available_memory < required_bytes) {
        throw std::runtime_error(
            "[ERROR]: Device out of memory. Tried to allocate " +
            human_readable_memory(required_bytes) +
            ", available: " + human_readable_memory(available_memory));
    }

    std::string shape = "";
    for (uint64_t& s : this->shape_) {
        shape += std::to_string(s) + " ";
    }

    std::cout << "Tensor " << this->name_ << " with shape: " << shape
              << "memory requirement: " << human_readable_memory(required_bytes)
              << std::endl;

    return std::shared_ptr<void>(malloc(required_bytes), free);
}

Tensor::Tensor(const SafeTensorEntry& tensor_entry, const Device& device) {
    std::ifstream r_handle(tensor_entry.filepath, std::ios::binary);

    if (!r_handle) {
        throw std::runtime_error("Could not open file: " +
                                 tensor_entry.filepath.string());
    }

    this->name_ = tensor_entry.name;
    this->dtype_ = tensor_entry.dtype;
    this->shape_ = tensor_entry.shape;
    this->device_ = device;

    uint64_t required_bytes =
        tensor_entry.offset_end - tensor_entry.offset_start;

    this->data_ = this->_get_memory(required_bytes, device);
    uint64_t header_size = 0;
    r_handle.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    r_handle.seekg(
        static_cast<std::streamoff>(tensor_entry.offset_start + header_size),
        std::ios::cur);  // skips the header and the tensor offset

    r_handle.read(this->data<char>(),
                  static_cast<std::streamsize>(required_bytes));

    r_handle.close();
}

Tensor::Tensor(const std::vector<uint64_t>& shape, const Device& device,
               const DType& dtype) {
    this->shape_ = shape;
    this->device_ = device;
    this->dtype_ = dtype;
    uint64_t num_el = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        num_el *= shape[i];
    }
    this->data_ = this->_get_memory(dtype_size(dtype) * num_el, device);
}

}  // namespace inferX
