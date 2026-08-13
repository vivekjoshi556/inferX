#include "utils.h"
#include "tensor.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include "device_context/device_context_registry.h"

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
                 .shape = el.value()["shape"].get<TensorShape>(),
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
        tensors.try_emplace(el.first, this->load_tensor(el.second, device));
    }

    return tensors;
}

TensorStorage::TensorStorage(const Device& device, const size_t req_mem,
                             const DType& dtype)
    : device_(device),
      nbytes_(req_mem),
      dtype_(dtype),
      d_ctx_(DeviceContextRegistry::instance().get(device)) {
    this->data_ = this->d_ctx_.allocate(req_mem);
}

Tensor SafeTensorFile::load_tensor(const SafeTensorEntry& tensor_entry,
                                   const Device& device) {
    std::ifstream r_handle(tensor_entry.filepath, std::ios::binary);

    if (!r_handle) {
        throw std::runtime_error("Could not open file: " +
                                 tensor_entry.filepath.string());
    }

    Tensor t(tensor_entry.shape, device, tensor_entry.dtype);
    uint64_t required_bytes =
        tensor_entry.offset_end - tensor_entry.offset_start;

    uint64_t header_size = 0;
    r_handle.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    if (device.type == DeviceType::CPU) {
        r_handle.read(t.data<char>(),
                      static_cast<std::streamsize>(required_bytes));
        return t;
    }

    size_t offset = 0;
    constexpr size_t CHUNK_SIZE = 64 * 1024 * 1024;  // 64 MB
    std::vector<char> buffer(CHUNK_SIZE);

    r_handle.seekg(
        static_cast<std::streamoff>(tensor_entry.offset_start + header_size),
        std::ios::cur);  // skips the header and the tensor offset

    while (offset < required_bytes) {
        size_t bytes_to_read = std::min(CHUNK_SIZE, required_bytes - offset);

        r_handle.read(buffer.data(), bytes_to_read);

        t.storage().copy_from_host(buffer.data(), bytes_to_read, offset);

        offset += bytes_to_read;
    }

    r_handle.close();
    return t;
}

Tensor::Tensor(const TensorShape& shape, const Device& device,
               const DType& dtype) {
    this->shape_ = shape;

    this->storage_ = std::make_shared<TensorStorage>(
        device,
        dtype_size(dtype) * std::accumulate(shape.begin(), shape.end(),
                                            uint64_t{1},
                                            std::multiplies<uint64_t>()),
        dtype);
}

}  // namespace inferX
