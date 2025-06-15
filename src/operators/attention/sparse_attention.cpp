#include "sparse_attention.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include <immintrin.h>

namespace deepcpp {
namespace operators {
namespace attention {

// SparseBlockMask Implementation
SparseBlockMask::SparseBlockMask(int64_t seq_len, int64_t block_size) 
    : seq_len(seq_len), block_size(block_size) {
    num_blocks = (seq_len + block_size - 1) / block_size;
    block_mask.resize(num_blocks, std::vector<bool>(num_blocks, false));
}

void SparseBlockMask::generate_local_pattern(int64_t window_size) {
    int64_t window_blocks = (window_size + block_size - 1) / block_size;
    
    for (int64_t i = 0; i < num_blocks; ++i) {
        for (int64_t j = 0; j < num_blocks; ++j) {
            if (std::abs(i - j) <= window_blocks) {
                block_mask[i][j] = true;
            }
        }
    }
}

void SparseBlockMask::generate_strided_pattern(int64_t stride) {
    int64_t stride_blocks = stride / block_size;
    
    for (int64_t i = 0; i < num_blocks; ++i) {
        for (int64_t j = 0; j < num_blocks; ++j) {
            if ((j - i) % stride_blocks == 0 && j >= i) {
                block_mask[i][j] = true;
            }
        }
    }
}

void SparseBlockMask::generate_bigbird_pattern(int64_t num_random_blocks, int64_t local_window) {
    // First add local pattern
    generate_local_pattern(local_window);
    
    // Add global attention for first and last blocks
    for (int64_t i = 0; i < num_blocks; ++i) {
        block_mask[i][0] = true;  // First block
        block_mask[0][i] = true;
        if (num_blocks > 1) {
            block_mask[i][num_blocks-1] = true;  // Last block
            block_mask[num_blocks-1][i] = true;
        }
    }
    
    // Add random connections
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(0, num_blocks - 1);
    
    for (int64_t i = 0; i < num_blocks; ++i) {
        for (int64_t r = 0; r < num_random_blocks; ++r) {
            int64_t random_j = dist(gen);
            block_mask[i][random_j] = true;
            block_mask[random_j][i] = true;  // Symmetric
        }
    }
}

int64_t SparseBlockMask::count_active_blocks() const {
    int64_t count = 0;
    for (const auto& row : block_mask) {
        for (bool active : row) {
            if (active) count++;
        }
    }
    return count;
}

float SparseBlockMask::sparsity_ratio() const {
    int64_t total_blocks = num_blocks * num_blocks;
    int64_t active_blocks = count_active_blocks();
    return 1.0f - static_cast<float>(active_blocks) / total_blocks;
}

// LocalAttention Implementation
LocalAttention::LocalAttention(const Config& config) : config_(config) {
    if (config_.scale < 0) {
        config_.scale = 1.0f / std::sqrt(64.0f);  // Default assuming head_dim=64
    }
}

Tensor LocalAttention::forward(const Tensor& q, const Tensor& k, const Tensor& v) {
    const auto& q_shape = q.shape();
    int64_t batch_size = q_shape[0];
    int64_t seq_len = q_shape[1];
    int64_t num_heads = q_shape[2];
    int64_t head_dim = q_shape[3];
    
    Tensor output(q_shape, q.dtype());
    
    local_attention_kernel(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, seq_len, num_heads, head_dim
    );
    
    return output;
}

void LocalAttention::local_attention_kernel(
    const float* q_ptr, const float* k_ptr, const float* v_ptr,
    float* output_ptr, int64_t batch_size, int64_t seq_len, 
    int64_t num_heads, int64_t head_dim) {
    
    float scale = config_.scale;
    int64_t half_window = config_.window_size / 2;
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t i = 0; i < seq_len; ++i) {
                
                // Determine local window bounds
                int64_t start_j = config_.bidirectional ? 
                                 std::max(0LL, i - half_window) : 
                                 std::max(0LL, i - config_.window_size + 1);
                int64_t end_j = std::min(seq_len, i + half_window + 1);
                
                // Find maximum score for numerical stability
                float max_score = -std::numeric_limits<float>::infinity();
                std::vector<float> scores(end_j - start_j);
                
                for (int64_t j = start_j; j < end_j; ++j) {
                    float score = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        int64_t q_idx = b * seq_len * num_heads * head_dim + 
                                       i * num_heads * head_dim + h * head_dim + d;
                        int64_t k_idx = b * seq_len * num_heads * head_dim + 
                                       j * num_heads * head_dim + h * head_dim + d;
                        score += q_ptr[q_idx] * k_ptr[k_idx];
                    }
                    score *= scale;
                    scores[j - start_j] = score;
                    max_score = std::max(max_score, score);
                }
                
                // Compute softmax
                float sum_exp = 0.0f;
                for (int64_t idx = 0; idx < scores.size(); ++idx) {
                    scores[idx] = std::exp(scores[idx] - max_score);
                    sum_exp += scores[idx];
                }
                
                // Compute output
                int64_t out_base = b * seq_len * num_heads * head_dim + 
                                  i * num_heads * head_dim + h * head_dim;
                
                for (int64_t d = 0; d < head_dim; ++d) {
                    output_ptr[out_base + d] = 0.0f;
                }
                
                for (int64_t j = start_j; j < end_j; ++j) {
                    float weight = scores[j - start_j] / sum_exp;
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

size_t LocalAttention::estimate_memory_usage(int64_t seq_len, int64_t head_dim) const {
    // O(window_size * head_dim) per position
    return config_.window_size * head_dim * sizeof(float) * 2;  // scores + workspace
}

// StridedAttention Implementation
StridedAttention::StridedAttention(const Config& config) : config_(config) {
    if (config_.scale < 0) {
        config_.scale = 1.0f / std::sqrt(64.0f);
    }
}

Tensor StridedAttention::forward(const Tensor& q, const Tensor& k, const Tensor& v) {
    const auto& q_shape = q.shape();
    int64_t batch_size = q_shape[0];
    int64_t seq_len = q_shape[1];
    int64_t num_heads = q_shape[2];
    int64_t head_dim = q_shape[3];
    
    Tensor output(q_shape, q.dtype());
    
    strided_attention_kernel(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, seq_len, num_heads, head_dim
    );
    
    return output;
}

void StridedAttention::strided_attention_kernel(
    const float* q_ptr, const float* k_ptr, const float* v_ptr,
    float* output_ptr, int64_t batch_size, int64_t seq_len,
    int64_t num_heads, int64_t head_dim) {
    
    float scale = config_.scale;
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t i = 0; i < seq_len; ++i) {
                
                std::vector<int64_t> attend_positions;
                
                // Add local window
                int64_t local_start = std::max(0LL, i - config_.local_window / 2);
                int64_t local_end = std::min(seq_len, i + config_.local_window / 2 + 1);
                for (int64_t j = local_start; j < local_end; ++j) {
                    attend_positions.push_back(j);
                }
                
                // Add strided positions
                for (int64_t j = 0; j < seq_len; j += config_.stride) {
                    if (j != i && std::find(attend_positions.begin(), attend_positions.end(), j) == attend_positions.end()) {
                        attend_positions.push_back(j);
                    }
                }
                
                if (attend_positions.empty()) continue;
                
                // Compute attention scores
                float max_score = -std::numeric_limits<float>::infinity();
                std::vector<float> scores(attend_positions.size());
                
                for (size_t idx = 0; idx < attend_positions.size(); ++idx) {
                    int64_t j = attend_positions[idx];
                    float score = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        int64_t q_idx = b * seq_len * num_heads * head_dim + 
                                       i * num_heads * head_dim + h * head_dim + d;
                        int64_t k_idx = b * seq_len * num_heads * head_dim + 
                                       j * num_heads * head_dim + h * head_dim + d;
                        score += q_ptr[q_idx] * k_ptr[k_idx];
                    }
                    score *= scale;
                    scores[idx] = score;
                    max_score = std::max(max_score, score);
                }
                
                // Softmax
                float sum_exp = 0.0f;
                for (size_t idx = 0; idx < scores.size(); ++idx) {
                    scores[idx] = std::exp(scores[idx] - max_score);
                    sum_exp += scores[idx];
                }
                
                // Output computation
                int64_t out_base = b * seq_len * num_heads * head_dim + 
                                  i * num_heads * head_dim + h * head_dim;
                
                for (int64_t d = 0; d < head_dim; ++d) {
                    output_ptr[out_base + d] = 0.0f;
                }
                
                for (size_t idx = 0; idx < attend_positions.size(); ++idx) {
                    int64_t j = attend_positions[idx];
                    float weight = scores[idx] / sum_exp;
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

// BigBirdAttention Implementation
BigBirdAttention::BigBirdAttention(const Config& config) : config_(config) {
    if (config_.scale < 0) {
        config_.scale = 1.0f / std::sqrt(64.0f);
    }
}

Tensor BigBirdAttention::forward(const Tensor& q, const Tensor& k, const Tensor& v) {
    const auto& q_shape = q.shape();
    int64_t batch_size = q_shape[0];
    int64_t seq_len = q_shape[1];
    int64_t num_heads = q_shape[2];
    int64_t head_dim = q_shape[3];
    
    // Get or create attention pattern
    if (cached_patterns_.find(seq_len) == cached_patterns_.end()) {
        precompute_attention_pattern(seq_len);
    }
    
    const auto& mask = cached_patterns_[seq_len];
    Tensor output(q_shape, q.dtype());
    
    bigbird_attention_kernel(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), mask, batch_size, seq_len, num_heads, head_dim
    );
    
    return output;
}

void BigBirdAttention::precompute_attention_pattern(int64_t seq_len) {
    SparseBlockMask mask(seq_len, config_.block_size);
    mask.generate_bigbird_pattern(config_.num_random_blocks, config_.window_size);
    cached_patterns_[seq_len] = std::move(mask);
}

void BigBirdAttention::bigbird_attention_kernel(
    const float* q_ptr, const float* k_ptr, const float* v_ptr,
    float* output_ptr, const SparseBlockMask& mask,
    int64_t batch_size, int64_t seq_len, int64_t num_heads, int64_t head_dim) {
    
    float scale = config_.scale;
    
    // Block-wise computation
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t block_i = 0; block_i < mask.num_blocks; ++block_i) {
                for (int64_t block_j = 0; block_j < mask.num_blocks; ++block_j) {
                    
                    if (!mask.block_mask[block_i][block_j]) continue;
                    
                    // Compute attention for this block pair
                    int64_t start_i = block_i * mask.block_size;
                    int64_t end_i = std::min(seq_len, (block_i + 1) * mask.block_size);
                    int64_t start_j = block_j * mask.block_size;
                    int64_t end_j = std::min(seq_len, (block_j + 1) * mask.block_size);
                    
                    for (int64_t i = start_i; i < end_i; ++i) {
                        
                        // Collect scores for this query position within active blocks
                        std::vector<std::pair<int64_t, float>> position_scores;
                        
                        for (int64_t active_block_j = 0; active_block_j < mask.num_blocks; ++active_block_j) {
                            if (!mask.block_mask[block_i][active_block_j]) continue;
                            
                            int64_t block_start_j = active_block_j * mask.block_size;
                            int64_t block_end_j = std::min(seq_len, (active_block_j + 1) * mask.block_size);
                            
                            for (int64_t j = block_start_j; j < block_end_j; ++j) {
                                float score = 0.0f;
                                for (int64_t d = 0; d < head_dim; ++d) {
                                    int64_t q_idx = b * seq_len * num_heads * head_dim + 
                                                   i * num_heads * head_dim + h * head_dim + d;
                                    int64_t k_idx = b * seq_len * num_heads * head_dim + 
                                                   j * num_heads * head_dim + h * head_dim + d;
                                    score += q_ptr[q_idx] * k_ptr[k_idx];
                                }
                                position_scores.emplace_back(j, score * scale);
                            }
                        }
                        
                        if (position_scores.empty()) continue;
                        
                        // Softmax over all attended positions
                        float max_score = -std::numeric_limits<float>::infinity();
                        for (const auto& pair : position_scores) {
                            max_score = std::max(max_score, pair.second);
                        }
                        
                        float sum_exp = 0.0f;
                        std::vector<float> weights(position_scores.size());
                        for (size_t idx = 0; idx < position_scores.size(); ++idx) {
                            weights[idx] = std::exp(position_scores[idx].second - max_score);
                            sum_exp += weights[idx];
                        }
                        
                        // Compute output
                        int64_t out_base = b * seq_len * num_heads * head_dim + 
                                          i * num_heads * head_dim + h * head_dim;
                        
                        for (int64_t d = 0; d < head_dim; ++d) {
                            float output_val = 0.0f;
                            for (size_t idx = 0; idx < position_scores.size(); ++idx) {
                                int64_t j = position_scores[idx].first;
                                float weight = weights[idx] / sum_exp;
                                int64_t v_idx = b * seq_len * num_heads * head_dim + 
                                               j * num_heads * head_dim + h * head_dim + d;
                                output_val += weight * v_ptr[v_idx];
                            }
                            output_ptr[out_base + d] = output_val;
                        }
                    }
                }
            }
        }
    }
}

// Utility functions
namespace sparse_utils {

SparseBlockMask create_local_pattern(int64_t seq_len, int64_t block_size, int64_t window_size) {
    SparseBlockMask mask(seq_len, block_size);
    mask.generate_local_pattern(window_size);
    return mask;
}

SparseBlockMask create_strided_pattern(int64_t seq_len, int64_t block_size, int64_t stride) {
    SparseBlockMask mask(seq_len, block_size);
    mask.generate_strided_pattern(stride);
    return mask;
}

SparseBlockMask create_bigbird_pattern(int64_t seq_len, int64_t block_size,
                                     int64_t num_random, int64_t window_size, int64_t seed) {
    SparseBlockMask mask(seq_len, block_size);
    mask.generate_bigbird_pattern(num_random, window_size);
    return mask;
}

float compute_sparsity_ratio(const SparseBlockMask& mask) {
    return mask.sparsity_ratio();
}

int64_t count_operations(const SparseBlockMask& mask, int64_t head_dim) {
    int64_t active_blocks = mask.count_active_blocks();
    return active_blocks * mask.block_size * mask.block_size * head_dim;
}

} // namespace sparse_utils

} // namespace attention
} // namespace operators
} // namespace deepcpp 