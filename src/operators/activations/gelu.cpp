#include "gelu.h"
#include <cmath>

namespace deepcpp {
namespace activations {

void gelu_forward(const float* input, float* output, int64_t size) {
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    
    for (int64_t i = 0; i < size; ++i) {
        float x = input[i];
        float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(tanh_arg));
    }
}

} // namespace activations
} // namespace deepcpp 