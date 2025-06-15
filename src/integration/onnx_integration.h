#pragma once

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>

// Include our components
#include "operators/attention/sparse_attention.h"
#include "operators/attention/linear_attention.h"
#include "operators/attention/multi_query_attention.h"
#include "operators/models/ssm.h"
#include "operators/models/mixture_of_experts.h"
#include "core/tensor/tensor.h"

namespace deepcpp {
namespace integration {

// Custom operator base class for ONNX Runtime integration
class CustomOperatorBase {
public:
    virtual ~CustomOperatorBase() = default;
    
    virtual const char* GetName() const = 0;
    virtual const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    
    virtual OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
        return INPUT_OUTPUT_REQUIRED;
    }
    
    virtual OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t index) const {
        return INPUT_OUTPUT_REQUIRED;
    }
    
    virtual size_t GetInputTypeCount() const = 0;
    virtual size_t GetOutputTypeCount() const = 0;
    virtual ONNXTensorElementDataType GetInputType(size_t index) const = 0;
    virtual ONNXTensorElementDataType GetOutputType(size_t index) const = 0;
    
    virtual void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const = 0;
    virtual void KernelCompute(void* op_kernel, OrtKernelContext* context) const = 0;
    virtual void KernelDestroy(void* op_kernel) const = 0;
};

// Sparse Attention Custom Operator
class SparseAttentionOp : public CustomOperatorBase {
private:
    std::string pattern_type_;
    int window_size_;
    int stride_;
    
public:
    SparseAttentionOp(const std::string& pattern_type, int window_size = 128, int stride = 4)
        : pattern_type_(pattern_type), window_size_(window_size), stride_(stride) {}
    
    const char* GetName() const override { return "SparseAttention"; }
    size_t GetInputTypeCount() const override { return 3; } // Q, K, V
    size_t GetOutputTypeCount() const override { return 1; } // Output
    
    ONNXTensorElementDataType GetInputType(size_t index) const override {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    ONNXTensorElementDataType GetOutputType(size_t index) const override {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const override;
    void KernelCompute(void* op_kernel, OrtKernelContext* context) const override;
    void KernelDestroy(void* op_kernel) const override;
};

// Linear Attention Custom Operator
class LinearAttentionOp : public CustomOperatorBase {
private:
    std::string kernel_type_;
    int num_features_;
    
public:
    LinearAttentionOp(const std::string& kernel_type, int num_features = 256)
        : kernel_type_(kernel_type), num_features_(num_features) {}
    
    const char* GetName() const override { return "LinearAttention"; }
    size_t GetInputTypeCount() const override { return 3; }
    size_t GetOutputTypeCount() const override { return 1; }
    
    ONNXTensorElementDataType GetInputType(size_t index) const override {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    ONNXTensorElementDataType GetOutputType(size_t index) const override {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const override;
    void KernelCompute(void* op_kernel, OrtKernelContext* context) const override;
    void KernelDestroy(void* op_kernel) const override;
};

// Multi-Query Attention Custom Operator
class MultiQueryAttentionOp : public CustomOperatorBase {
private:
    int num_kv_heads_;
    
public:
    MultiQueryAttentionOp(int num_kv_heads) : num_kv_heads_(num_kv_heads) {}
    
    const char* GetName() const override { return "MultiQueryAttention"; }
    size_t GetInputTypeCount() const override { return 3; }
    size_t GetOutputTypeCount() const override { return 1; }
    
    ONNXTensorElementDataType GetInputType(size_t index) const override {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    ONNXTensorElementDataType GetOutputType(size_t index) const override {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const override;
    void KernelCompute(void* op_kernel, OrtKernelContext* context) const override;
    void KernelDestroy(void* op_kernel) const override;
};

// State Space Model Custom Operator
class SSMOp : public CustomOperatorBase {
private:
    std::string ssm_type_;
    int state_size_;
    
public:
    SSMOp(const std::string& ssm_type, int state_size) 
        : ssm_type_(ssm_type), state_size_(state_size) {}
    
    const char* GetName() const override { return "StateSpaceModel"; }
    size_t GetInputTypeCount() const override { return 1; }
    size_t GetOutputTypeCount() const override { return 1; }
    
    ONNXTensorElementDataType GetInputType(size_t index) const override {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    ONNXTensorElementDataType GetOutputType(size_t index) const override {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const override;
    void KernelCompute(void* op_kernel, OrtKernelContext* context) const override;
    void KernelDestroy(void* op_kernel) const override;
};

// Mixture of Experts Custom Operator
class MixtureOfExpertsOp : public CustomOperatorBase {
private:
    int num_experts_;
    int top_k_;
    
public:
    MixtureOfExpertsOp(int num_experts, int top_k) 
        : num_experts_(num_experts), top_k_(top_k) {}
    
    const char* GetName() const override { return "MixtureOfExperts"; }
    size_t GetInputTypeCount() const override { return 1; }
    size_t GetOutputTypeCount() const override { return 1; }
    
    ONNXTensorElementDataType GetInputType(size_t index) const override {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    ONNXTensorElementDataType GetOutputType(size_t index) const override {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const override;
    void KernelCompute(void* op_kernel, OrtKernelContext* context) const override;
    void KernelDestroy(void* op_kernel) const override;
};

// Integration Manager
class ONNXIntegrationManager {
private:
    std::vector<std::unique_ptr<CustomOperatorBase>> custom_ops_;
    
public:
    ONNXIntegrationManager();
    ~ONNXIntegrationManager() = default;
    
    // Register all our custom operators
    void register_all_operators();
    
    // Register specific operators
    void register_sparse_attention_ops();
    void register_linear_attention_ops();
    void register_multi_query_attention_ops();
    void register_ssm_ops();
    void register_moe_ops();
    
    // ONNX Runtime integration
    OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
    
    // Utility functions
    static std::shared_ptr<core::Tensor> ort_tensor_to_deepcpp_tensor(const OrtValue* ort_tensor);
    static OrtValue* deepcpp_tensor_to_ort_tensor(std::shared_ptr<core::Tensor> tensor, 
                                                  const OrtApi& api, OrtAllocator* allocator);
    
    // Model export utilities
    void export_model_with_custom_ops(const std::string& model_path, 
                                     const std::string& output_path);
    
    // Performance testing with ONNX models
    void benchmark_onnx_model_with_custom_ops(const std::string& model_path,
                                             const std::vector<std::vector<int64_t>>& input_shapes);
};

// Utility functions for tensor conversion
namespace utils {
    std::vector<int64_t> get_tensor_shape(const OrtValue* tensor, const OrtApi& api);
    size_t get_tensor_size(const std::vector<int64_t>& shape);
    void copy_tensor_data(const OrtValue* src, OrtValue* dst, const OrtApi& api);
    
    // Performance profiling utilities
    class ONNXProfiler {
    private:
        std::map<std::string, std::vector<double>> op_timings_;
        
    public:
        void start_timing(const std::string& op_name);
        void end_timing(const std::string& op_name);
        void print_profile_report();
        void save_profile_report(const std::string& filename);
    };
}

// Factory functions for creating custom operators
std::unique_ptr<CustomOperatorBase> create_sparse_attention_op(const std::string& pattern_type);
std::unique_ptr<CustomOperatorBase> create_linear_attention_op(const std::string& kernel_type);
std::unique_ptr<CustomOperatorBase> create_multi_query_attention_op(int num_kv_heads);
std::unique_ptr<CustomOperatorBase> create_ssm_op(const std::string& ssm_type, int state_size);
std::unique_ptr<CustomOperatorBase> create_moe_op(int num_experts, int top_k);

} // namespace integration
} // namespace deepcpp

// C API for ONNX Runtime registration
extern "C" {
    OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
} 