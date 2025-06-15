#pragma once

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <immintrin.h>  // AVX/SSE
#include <omp.h>        // OpenMP

namespace deepcpp {
namespace core {

// Forward declarations
class Device;
class MemoryPool;
class ComputeStream;

// Data types supported by the framework
enum class DataType {
    FLOAT32,
    FLOAT16,
    BFLOAT16,
    INT64,
    INT32,
    INT16,
    INT8,
    UINT8,
    BOOL,
    COMPLEX64,
    COMPLEX128
};

// Memory layout strategies
enum class MemoryLayout {
    CONTIGUOUS,      // C-style contiguous
    FORTRAN,         // Fortran-style contiguous  
    STRIDED,         // Custom strides
    CHANNELS_LAST,   // NHWC format
    CHANNELS_FIRST   // NCHW format
};

// Device types for computation
enum class DeviceType {
    CPU,
    CUDA,
    OPENCL,
    METAL,
    VULKAN
};

// Tensor storage implementation
class TensorStorage {
public:
    TensorStorage(size_t size_bytes, DataType dtype, DeviceType device);
    ~TensorStorage();
    
    // Data access
    void* data_ptr() const { return data_; }
    size_t size_bytes() const { return size_bytes_; }
    size_t ref_count() const { return ref_count_; }
    
    // Reference counting
    void retain() { ++ref_count_; }
    void release() { if (--ref_count_ == 0) delete this; }
    
    // Device operations
    void to_device(DeviceType target_device);
    void synchronize();
    
    // Memory operations
    void copy_from(const TensorStorage& other);
    void fill(const void* value);
    void zero();
    
private:
    void* data_;
    size_t size_bytes_;
    std::atomic<size_t> ref_count_;
    DataType dtype_;
    DeviceType device_;
    std::unique_ptr<MemoryPool> memory_pool_;
};

// Shape and stride information
class TensorShape {
public:
    TensorShape() = default;
    TensorShape(std::initializer_list<int64_t> dims);
    TensorShape(const std::vector<int64_t>& dims);
    
    // Shape access
    size_t ndim() const { return dims_.size(); }
    int64_t size(int dim) const;
    int64_t numel() const;
    const std::vector<int64_t>& dims() const { return dims_; }
    
    // Shape operations
    TensorShape squeeze(int dim = -1) const;
    TensorShape unsqueeze(int dim) const;
    TensorShape transpose(int dim0, int dim1) const;
    TensorShape permute(const std::vector<int>& dims) const;
    TensorShape reshape(const std::vector<int64_t>& new_shape) const;
    TensorShape expand(const std::vector<int64_t>& sizes) const;
    
    // Broadcasting
    static TensorShape broadcast_shapes(const TensorShape& a, const TensorShape& b);
    bool is_broadcastable_with(const TensorShape& other) const;
    
    // Comparison
    bool operator==(const TensorShape& other) const;
    bool operator!=(const TensorShape& other) const;
    
    // String representation
    std::string to_string() const;
    
private:
    std::vector<int64_t> dims_;
    mutable int64_t cached_numel_ = -1;
};

// Stride calculation and management
class TensorStrides {
public:
    TensorStrides() = default;
    TensorStrides(const TensorShape& shape, MemoryLayout layout = MemoryLayout::CONTIGUOUS);
    TensorStrides(const std::vector<int64_t>& strides);
    
    // Stride access
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t stride(int dim) const;
    
    // Layout properties
    bool is_contiguous() const;
    bool is_channels_last() const;
    bool is_fortran_contiguous() const;
    
    // Operations
    TensorStrides transpose(int dim0, int dim1) const;
    TensorStrides permute(const std::vector<int>& dims) const;
    TensorStrides squeeze(int dim) const;
    TensorStrides unsqueeze(int dim) const;
    
    // Memory calculations
    size_t storage_offset(const std::vector<int64_t>& indices) const;
    std::pair<size_t, size_t> contiguous_range(const TensorShape& shape) const;
    
private:
    std::vector<int64_t> strides_;
    MemoryLayout layout_ = MemoryLayout::CONTIGUOUS;
};

// Main Tensor class with comprehensive functionality
class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int64_t>& shape, DataType dtype = DataType::FLOAT32);
    
    // Properties
    const std::vector<int64_t>& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    size_t ndim() const { return shape_.size(); }
    int64_t size(int dim) const;
    int64_t numel() const;
    
    // Data access
    void* data_ptr() const;
    template<typename T> T* data_ptr() const;
    
    // Operations
    Tensor add(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor mm(const Tensor& other) const;  // Matrix multiply
    
    // Static creation functions
    static Tensor zeros(const std::vector<int64_t>& shape);
    static Tensor ones(const std::vector<int64_t>& shape);
    static Tensor randn(const std::vector<int64_t>& shape);
    
private:
    std::vector<int64_t> shape_;
    DataType dtype_;
    std::shared_ptr<void> data_;
    size_t storage_bytes_;
};

// Utility functions
size_t get_element_size(DataType dtype);
std::string dtype_to_string(DataType dtype);
std::string device_to_string(DeviceType device);

// Type traits for template specializations
template<typename T> 
constexpr DataType get_data_type();

template<> constexpr DataType get_data_type<float>() { return DataType::FLOAT32; }
template<> constexpr DataType get_data_type<double>() { return DataType::COMPLEX128; }
template<> constexpr DataType get_data_type<int64_t>() { return DataType::INT64; }
template<> constexpr DataType get_data_type<int32_t>() { return DataType::INT32; }
template<> constexpr DataType get_data_type<int16_t>() { return DataType::INT16; }
template<> constexpr DataType get_data_type<int8_t>() { return DataType::INT8; }
template<> constexpr DataType get_data_type<uint8_t>() { return DataType::UINT8; }
template<> constexpr DataType get_data_type<bool>() { return DataType::BOOL; }

} // namespace core
} // namespace deepcpp 