#pragma once

#include "../../core/tensor/tensor.h"

namespace deepcpp {
namespace operators {
namespace activations {

using deepcpp::core::Tensor;

class SwishActivation {
public:
    SwishActivation(float beta = 1.0f);
    
    Tensor forward(const Tensor& input);
    
private:
    float beta_;
    
    void swish_kernel(const float* input, float* output, int64_t size);
};

} // namespace activations
} // namespace operators
} // namespace deepcpp 