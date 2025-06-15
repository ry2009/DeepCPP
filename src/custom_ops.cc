#include "custom_ops.h"
#include <iostream>
#include <cmath>

// SSM Scan kernel constructor
SSMScanKernel::SSMScanKernel(const OrtApi& api, const OrtKernelInfo* info)
    : ort_api_(api), selective_scan_mode_(true), dt_min_(0.001f), dt_max_(0.1f) {
    // Could read attributes from OrtKernelInfo if needed
    // For now, use default values
}

// SSM Scan kernel computation
void SSMScanKernel::Compute(OrtKernelContext* context) {
    // Use the C++ wrapper for easier API access
    Ort::KernelContext ctx(context);
    
    // Get input tensors
    auto input_x = ctx.GetInput(0);
    auto input_delta = ctx.GetInput(1);
    auto input_A = ctx.GetInput(2);
    auto input_B = ctx.GetInput(3);
    
    // Get input tensor info
    auto input_info = input_x.GetTensorTypeAndShapeInfo();
    auto input_shape = input_info.GetShape();
    
    // For now, implement a simple pass-through (placeholder)
    // Real SSM scan implementation would go here
    
    // Create output tensor with same shape as input
    auto output = ctx.GetOutput(0, input_shape.data(), input_shape.size());
    
    // Get data pointers
    const float* x_data = input_x.GetTensorData<float>();
    float* output_data = output.GetTensorMutableData<float>();
    
    // Calculate total elements
    int64_t total_elements = 1;
    for (auto dim : input_shape) {
        total_elements *= dim;
    }
    
    // Simple placeholder: copy input to output
    // TODO: Replace with actual SSM scan computation
    std::memcpy(output_data, x_data, total_elements * sizeof(float));
}

// SSMScan op factory method
void* SSMScan::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new SSMScanKernel(api, info);
}

// Registration function
extern "C" {

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
    Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);
    
    static SSMScan ssm_scan_op;
    
    Ort::CustomOpDomain domain("ryan_ops");
    domain.Add(&ssm_scan_op);
    
    Ort::SessionOptions sess_opts(options);
    sess_opts.Add(domain);
    
    return nullptr;
}

} 