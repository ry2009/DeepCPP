#include "geglu.h"
#include <cmath>

namespace deepcpp {
namespace operators {
namespace activations {

Tensor GEGLUActivation::forward(const Tensor& input) {
    // Placeholder implementation
    return input;
}

void GEGLUActivation::geglu_kernel(const float* input, float* output, int64_t size, int64_t hidden_dim) {
    // GEGLU: x * GELU(y) where input is split into x and y
    for (int64_t i = 0; i < size / 2; ++i) {
        float x = input[i];
        float y = input[i + hidden_dim];
        // GELU approximation
        float gelu_y = 0.5f * y * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (y + 0.044715f * y * y * y)));
        output[i] = x * gelu_y;
    }
}

} // namespace activations
} // namespace operators
} // namespace deepcpp 