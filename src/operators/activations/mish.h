#pragma once

#include "../../core/tensor/tensor.h"

namespace deepcpp {
namespace operators {
namespace activations {

using deepcpp::core::Tensor;

class MishActivation {
public:
    MishActivation() = default;
    
    Tensor forward(const Tensor& input);
    
private:
    void mish_kernel(const float* input, float* output, int64_t size);
};

} // namespace activations
} // namespace operators
} // namespace deepcpp 