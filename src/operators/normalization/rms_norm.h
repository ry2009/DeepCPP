#pragma once

#include "../../core/tensor/tensor.h"

namespace deepcpp {
namespace operators {
namespace normalization {

using deepcpp::core::Tensor;

class RMSNorm {
public:
    RMSNorm(int64_t hidden_size, float epsilon = 1e-5f);
    
    Tensor forward(const Tensor& input);
    
    void set_weight(const Tensor& weight);
    
private:
    int64_t hidden_size_;
    float epsilon_;
    std::unique_ptr<Tensor> weight_;
    
    void rms_norm_kernel(const float* input, const float* weight, 
                        float* output, int64_t batch_size, int64_t seq_len, 
                        int64_t hidden_size, float epsilon);
};

} // namespace normalization
} // namespace operators
} // namespace deepcpp 