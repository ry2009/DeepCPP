#pragma once

#include <cstdint>

namespace deepcpp {
namespace activations {

/**
 * GELU (Gaussian Error Linear Unit) activation function
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 */
void gelu_forward(const float* input, float* output, int64_t size);

} // namespace activations
} // namespace deepcpp 