#pragma once

#include "../../core/tensor/tensor.h"

namespace deepcpp {
namespace operators {
namespace activations {

using deepcpp::core::Tensor;

class GEGLUActivation {
public:
    GEGLUActivation() = default;
    
    Tensor forward(const Tensor& input);
    
private:
    void geglu_kernel(const float* input, float* output, int64_t size, int64_t hidden_dim);
};

} // namespace activations
} // namespace operators
} // namespace deepcpp 