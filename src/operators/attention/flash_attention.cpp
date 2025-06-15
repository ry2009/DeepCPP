#include "flash_attention.h"
#include "../../core/memory/memory_pool.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// For now, using the core namespace for Tensor
using deepcpp::core::Tensor;

namespace deepcpp {
namespace operators {
namespace attention {

FlashAttention::FlashAttention(const Config& config) : config_(config) {
    // Set default scale if not provided
    if (config_.scale < 0) {
        config_.scale = 1.0f / std::sqrt(64.0f); // Default head_dim
    }
    
    // Initialize workspace
    workspace_size_ = 1024 * 1024; // 1MB default workspace
    workspace_ = std::make_unique<float[]>(workspace_size_ / sizeof(float));
}

FlashAttention::~FlashAttention() = default;

Tensor FlashAttention::forward(const Tensor& q, const Tensor& k, const Tensor& v, 
                              const Tensor& mask) {
    const auto& q_shape = q.shape();
    int64_t batch_size = q_shape[0];
    int64_t seq_len = q_shape[1];
    int64_t num_heads = q_shape[2];
    int64_t head_dim = q_shape[3];
    
    // Create output tensor
    Tensor output(q_shape, q.dtype());
    
    // Create LSE tensor (log-sum-exp) for numerical stability
    std::vector<int64_t> lse_shape = {batch_size, num_heads, seq_len};
    Tensor lse(lse_shape, DataType::FLOAT32);
    
    // Call the optimized kernel
    flash_attention_forward_kernel(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), lse.data_ptr<float>(),
        batch_size, seq_len, num_heads, head_dim,
        mask.numel() > 0 ? mask.data_ptr<float>() : nullptr
    );
    
    return output;
}

void FlashAttention::flash_attention_forward_kernel(
    const float* q_ptr, const float* k_ptr, const float* v_ptr,
    float* output_ptr, float* lse_ptr,
    int64_t batch_size, int64_t seq_len, int64_t num_heads, int64_t head_dim,
    const float* mask_ptr) {
    
    float scale = config_.scale;
    int64_t block_size = config_.block_size_q;
    
    // Simplified implementation for now
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t i = 0; i < seq_len; ++i) {
                float max_score = -std::numeric_limits<float>::infinity();
                
                // First pass: find maximum score for numerical stability
                for (int64_t j = 0; j < seq_len; ++j) {
                    float score = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        int64_t q_idx = b * seq_len * num_heads * head_dim + 
                                       i * num_heads * head_dim + h * head_dim + d;
                        int64_t k_idx = b * seq_len * num_heads * head_dim + 
                                       j * num_heads * head_dim + h * head_dim + d;
                        score += q_ptr[q_idx] * k_ptr[k_idx];
                    }
                    score *= scale;
                    
                    // Apply mask if provided
                    if (mask_ptr) {
                        int64_t mask_idx = b * seq_len * seq_len + i * seq_len + j;
                        score += mask_ptr[mask_idx];
                    }
                    
                    // Apply causal mask if enabled
                    if (config_.use_causal_mask && j > i) {
                        score = -std::numeric_limits<float>::infinity();
                    }
                    
                    max_score = std::max(max_score, score);
                }
                
                // Second pass: compute softmax and output
                float sum_exp = 0.0f;
                std::vector<float> scores(seq_len);
                
                for (int64_t j = 0; j < seq_len; ++j) {
                    float score = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        int64_t q_idx = b * seq_len * num_heads * head_dim + 
                                       i * num_heads * head_dim + h * head_dim + d;
                        int64_t k_idx = b * seq_len * num_heads * head_dim + 
                                       j * num_heads * head_dim + h * head_dim + d;
                        score += q_ptr[q_idx] * k_ptr[k_idx];
                    }
                    score *= scale;
                    
                    // Apply mask if provided
                    if (mask_ptr) {
                        int64_t mask_idx = b * seq_len * seq_len + i * seq_len + j;
                        score += mask_ptr[mask_idx];
                    }
                    
                    // Apply causal mask if enabled
                    if (config_.use_causal_mask && j > i) {
                        score = -std::numeric_limits<float>::infinity();
                    }
                    
                    scores[j] = std::exp(score - max_score);
                    sum_exp += scores[j];
                }
                
                // Store LSE for backward pass
                int64_t lse_idx = b * num_heads * seq_len + h * seq_len + i;
                lse_ptr[lse_idx] = max_score + std::log(sum_exp);
                
                // Compute output
                int64_t out_base = b * seq_len * num_heads * head_dim + 
                                  i * num_heads * head_dim + h * head_dim;
                
                // Initialize output to zero
                for (int64_t d = 0; d < head_dim; ++d) {
                    output_ptr[out_base + d] = 0.0f;
                }
                
                // Weighted sum of values
                for (int64_t j = 0; j < seq_len; ++j) {
                    float weight = scores[j] / sum_exp;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        int64_t v_idx = b * seq_len * num_heads * head_dim + 
                                       j * num_heads * head_dim + h * head_dim + d;
                        output_ptr[out_base + d] += weight * v_ptr[v_idx];
                    }
                }
            }
        }
    }
}

size_t FlashAttention::estimate_memory_usage(int64_t batch_size, int64_t seq_len, 
                                            int64_t num_heads, int64_t head_dim) const {
    // Estimate memory usage for Flash Attention
    size_t input_size = batch_size * seq_len * num_heads * head_dim * sizeof(float) * 3; // Q, K, V
    size_t output_size = batch_size * seq_len * num_heads * head_dim * sizeof(float);
    size_t workspace_size = config_.block_size_q * config_.block_size_kv * sizeof(float) * 4;
    
    return input_size + output_size + workspace_size;
}

} // namespace attention
} // namespace operators
} // namespace deepcpp 