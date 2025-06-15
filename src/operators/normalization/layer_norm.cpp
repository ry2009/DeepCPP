#include "layer_norm.h"
#include <cmath>

namespace deepcpp {
namespace normalization {

void layer_norm_forward(const float* input, const float* weight, const float* bias,
                       float* output, int64_t batch_size, int64_t hidden_size, float eps) {
    
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* input_row = input + b * hidden_size;
        float* output_row = output + b * hidden_size;
        
        // Compute mean
        float mean = 0.0f;
        for (int64_t i = 0; i < hidden_size; ++i) {
            mean += input_row[i];
        }
        mean /= hidden_size;
        
        // Compute variance
        float variance = 0.0f;
        for (int64_t i = 0; i < hidden_size; ++i) {
            float diff = input_row[i] - mean;
            variance += diff * diff;
        }
        variance /= hidden_size;
        
        // Normalize
        float inv_std = 1.0f / std::sqrt(variance + eps);
        for (int64_t i = 0; i < hidden_size; ++i) {
            float normalized = (input_row[i] - mean) * inv_std;
            output_row[i] = normalized * weight[i] + bias[i];
        }
    }
}

} // namespace normalization
} // namespace deepcpp 