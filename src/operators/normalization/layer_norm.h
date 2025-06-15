#pragma once

#include <cstdint>

namespace deepcpp {
namespace normalization {

/**
 * Layer Normalization
 * Normalizes across the last dimension of the input tensor
 */
void layer_norm_forward(const float* input, const float* weight, const float* bias,
                       float* output, int64_t batch_size, int64_t hidden_size, 
                       float eps = 1e-5f);

} // namespace normalization
} // namespace deepcpp 