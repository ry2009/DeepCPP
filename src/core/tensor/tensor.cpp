#include "tensor.h"
#include <stdexcept>
#include <cstring>
#include <cstdlib>

namespace deepcpp {
namespace core {

// Utility function to get element size for different data types
size_t get_element_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return sizeof(float);
        case DataType::FLOAT16: return sizeof(uint16_t);
        case DataType::BFLOAT16: return sizeof(uint16_t);
        case DataType::INT64: return sizeof(int64_t);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::INT16: return sizeof(int16_t);
        case DataType::INT8: return sizeof(int8_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BOOL: return sizeof(bool);
        case DataType::COMPLEX64: return sizeof(float) * 2;
        case DataType::COMPLEX128: return sizeof(double) * 2;
        default: return sizeof(float);
    }
}

// Default constructor
Tensor::Tensor() : dtype_(DataType::FLOAT32), storage_bytes_(0) {}

// Constructor with shape and data type
Tensor::Tensor(const std::vector<int64_t>& shape, DataType dtype) 
    : shape_(shape), dtype_(dtype) {
    
    // Calculate total elements
    int64_t total_elements = 1;
    for (auto dim : shape) {
        total_elements *= dim;
    }
    
    // Calculate storage size
    storage_bytes_ = total_elements * get_element_size(dtype);
    
    // Allocate memory using shared_ptr
    data_ = std::shared_ptr<void>(std::malloc(storage_bytes_), std::free);
    if (!data_) {
        throw std::bad_alloc();
    }
    
    // Zero initialize
    std::memset(data_.get(), 0, storage_bytes_);
}

int64_t Tensor::size(int dim) const {
    if (dim < 0 || dim >= static_cast<int64_t>(shape_.size())) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape_[dim];
}

int64_t Tensor::numel() const {
    int64_t total = 1;
    for (auto dim : shape_) {
        total *= dim;
    }
    return total;
}

void* Tensor::data_ptr() const {
    return data_.get();
}

template<typename T>
T* Tensor::data_ptr() const {
    return static_cast<T*>(data_.get());
}

// Explicit template instantiations
template float* Tensor::data_ptr<float>() const;
template double* Tensor::data_ptr<double>() const;
template int32_t* Tensor::data_ptr<int32_t>() const;
template int64_t* Tensor::data_ptr<int64_t>() const;

// Static creation functions
Tensor Tensor::zeros(const std::vector<int64_t>& shape) {
    return Tensor(shape, DataType::FLOAT32);
}

Tensor Tensor::ones(const std::vector<int64_t>& shape) {
    Tensor tensor(shape, DataType::FLOAT32);
    float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        data[i] = 1.0f;
    }
    return tensor;
}

Tensor Tensor::randn(const std::vector<int64_t>& shape) {
    Tensor tensor(shape, DataType::FLOAT32);
    float* data = tensor.data_ptr<float>();
    
    // Simple random number generation (not cryptographically secure)
    for (int64_t i = 0; i < tensor.numel(); ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    return tensor;
}

// String utility functions
std::string dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "float32";
        case DataType::FLOAT16: return "float16";
        case DataType::BFLOAT16: return "bfloat16";
        case DataType::INT64: return "int64";
        case DataType::INT32: return "int32";
        case DataType::INT16: return "int16";
        case DataType::INT8: return "int8";
        case DataType::UINT8: return "uint8";
        case DataType::BOOL: return "bool";
        case DataType::COMPLEX64: return "complex64";
        case DataType::COMPLEX128: return "complex128";
        default: return "unknown";
    }
}

std::string device_to_string(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda";
        case DeviceType::OPENCL: return "opencl";
        case DeviceType::METAL: return "metal";
        case DeviceType::VULKAN: return "vulkan";
        default: return "unknown";
    }
}

} // namespace core
} // namespace deepcpp 