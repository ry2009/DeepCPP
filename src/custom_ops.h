#pragma once

#include "onnxruntime_cxx_api.h"
#include <memory>
#include <vector>

// Forward declarations
struct SSMScanKernel;

// Custom operation for SSM (State Space Model) selective scan
struct SSMScan final : Ort::CustomOpBase<SSMScan, SSMScanKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;
    const char* GetName() const { return "SSMScan"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    
    size_t GetInputTypeCount() const { return 4; }  // input, delta, A, B, C
    ONNXTensorElementDataType GetInputType(size_t index) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t index) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
        return INPUT_OUTPUT_REQUIRED;
    }
    
    OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t index) const {
        return INPUT_OUTPUT_REQUIRED;
    }
};

// Kernel implementation for SSM scan
struct SSMScanKernel {
    SSMScanKernel(const OrtApi& api, const OrtKernelInfo* info);
    
    void Compute(OrtKernelContext* context);
    
private:
    const OrtApi& ort_api_;
    
    // Configuration parameters (could be made configurable via attributes)
    bool selective_scan_mode_;
    float dt_min_;
    float dt_max_;
};

// Export function for registering custom ops
extern "C" {
    OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
} 