#include "linear_attention.h"
#include <cmath>
#include <random>
#include <algorithm>

namespace deepcpp {
namespace operators {
namespace attention {

// PerformerAttention Implementation
PerformerAttention::PerformerAttention(const Config& config) : LinearAttentionBase(config), rng_(42) {
    if (config_.scale < 0) {
        config_.scale = 1.0f / std::sqrt(64.0f); // Default head_dim
    }
}

Tensor PerformerAttention::forward(const Tensor& q, const Tensor& k, const Tensor& v) {
    const auto& q_shape = q.shape();
    int64_t batch_size = q_shape[0];
    int64_t seq_len = q_shape[1];
    int64_t num_heads = q_shape[2];
    int64_t head_dim = q_shape[3];
    
    // Initialize random features if needed
    if (random_features_.empty()) {
        initialize_random_features(num_heads, head_dim);
    }
    
    Tensor output(q_shape, q.dtype());
    
    // Create feature maps for Q and K
    int64_t num_features = config_.num_features;
    std::vector<int64_t> feature_shape = {batch_size, seq_len, num_heads, num_features};
    Tensor q_features(feature_shape, q.dtype());
    Tensor k_features(feature_shape, k.dtype());
    
    // Compute feature maps
    compute_feature_maps(
        q.data_ptr<float>(), q_features.data_ptr<float>(),
        batch_size, seq_len, num_heads, head_dim, random_features_, true
    );
    
    compute_feature_maps(
        k.data_ptr<float>(), k_features.data_ptr<float>(),
        batch_size, seq_len, num_heads, head_dim, random_features_, false
    );
    
    // Apply linear attention kernel
    if (config_.causal) {
        causal_linear_attention_kernel(
            q_features.data_ptr<float>(), k_features.data_ptr<float>(), v.data_ptr<float>(),
            output.data_ptr<float>(), batch_size, seq_len, num_heads, head_dim, num_features
        );
    } else {
        linear_attention_kernel(
            q_features.data_ptr<float>(), k_features.data_ptr<float>(), v.data_ptr<float>(),
            output.data_ptr<float>(), batch_size, seq_len, num_heads, head_dim, num_features
        );
    }
    
    return output;
}

void PerformerAttention::initialize_random_features(int64_t num_heads, int64_t head_dim) {
    random_features_.resize(num_heads);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int64_t h = 0; h < num_heads; ++h) {
        random_features_[h].resize(config_.num_features * head_dim);
        for (int64_t i = 0; i < config_.num_features * head_dim; ++i) {
            random_features_[h][i] = dist(rng_);
        }
    }
}

void PerformerAttention::compute_feature_maps(
    const float* input_ptr, float* feature_ptr,
    int64_t batch_size, int64_t seq_len, int64_t num_heads, int64_t head_dim,
    const std::vector<std::vector<float>>& features, bool is_query) {
    
    float normalization = std::sqrt(static_cast<float>(config_.num_features));
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            for (int64_t h = 0; h < num_heads; ++h) {
                for (int64_t f = 0; f < config_.num_features; ++f) {
                    
                    // Compute dot product with random feature
                    float dot_product = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        int64_t input_idx = b * seq_len * num_heads * head_dim + 
                                           s * num_heads * head_dim + h * head_dim + d;
                        int64_t feature_idx = f * head_dim + d;
                        dot_product += input_ptr[input_idx] * features[h][feature_idx];
                    }
                    
                    // Apply FAVOR+ kernel: exp(dot_product - ||x||²/2)
                    float norm_sq = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        int64_t input_idx = b * seq_len * num_heads * head_dim + 
                                           s * num_heads * head_dim + h * head_dim + d;
                        norm_sq += input_ptr[input_idx] * input_ptr[input_idx];
                    }
                    
                    float kernel_value = std::exp(dot_product - norm_sq * 0.5f) / normalization;
                    
                    int64_t output_idx = b * seq_len * num_heads * config_.num_features + 
                                        s * num_heads * config_.num_features + 
                                        h * config_.num_features + f;
                    feature_ptr[output_idx] = kernel_value;
                }
            }
        }
    }
}

void PerformerAttention::linear_attention_kernel(
    const float* q_features, const float* k_features, const float* v_ptr,
    float* output_ptr, int64_t batch_size, int64_t seq_len,
    int64_t num_heads, int64_t head_dim, int64_t num_features) {
    
    // Linear attention: O = Q_phi * (K_phi^T * V)
    // First compute K_phi^T * V for each head
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            
            // Compute K^T * V: [num_features, head_dim]
            std::vector<float> kv_matrix(num_features * head_dim, 0.0f);
            
            for (int64_t f = 0; f < num_features; ++f) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (int64_t s = 0; s < seq_len; ++s) {
                        int64_t k_idx = b * seq_len * num_heads * num_features + 
                                       s * num_heads * num_features + h * num_features + f;
                        int64_t v_idx = b * seq_len * num_heads * head_dim + 
                                       s * num_heads * head_dim + h * head_dim + d;
                        sum += k_features[k_idx] * v_ptr[v_idx];
                    }
                    kv_matrix[f * head_dim + d] = sum;
                }
            }
            
            // Now compute Q * (K^T * V)
            for (int64_t s = 0; s < seq_len; ++s) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (int64_t f = 0; f < num_features; ++f) {
                        int64_t q_idx = b * seq_len * num_heads * num_features + 
                                       s * num_heads * num_features + h * num_features + f;
                        sum += q_features[q_idx] * kv_matrix[f * head_dim + d];
                    }
                    
                    int64_t out_idx = b * seq_len * num_heads * head_dim + 
                                     s * num_heads * head_dim + h * head_dim + d;
                    output_ptr[out_idx] = sum * config_.scale;
                }
            }
        }
    }
}

void PerformerAttention::causal_linear_attention_kernel(
    const float* q_features, const float* k_features, const float* v_ptr,
    float* output_ptr, int64_t batch_size, int64_t seq_len,
    int64_t num_heads, int64_t head_dim, int64_t num_features) {
    
    // Causal linear attention: maintain running sum of K^T * V
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            
            // Running K^T * V matrix
            std::vector<float> running_kv(num_features * head_dim, 0.0f);
            
            for (int64_t s = 0; s < seq_len; ++s) {
                
                // Compute output for position s using current running_kv
                for (int64_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (int64_t f = 0; f < num_features; ++f) {
                        int64_t q_idx = b * seq_len * num_heads * num_features + 
                                       s * num_heads * num_features + h * num_features + f;
                        sum += q_features[q_idx] * running_kv[f * head_dim + d];
                    }
                    
                    int64_t out_idx = b * seq_len * num_heads * head_dim + 
                                     s * num_heads * head_dim + h * head_dim + d;
                    output_ptr[out_idx] = sum * config_.scale;
                }
                
                // Update running K^T * V with current position
                for (int64_t f = 0; f < num_features; ++f) {
                    for (int64_t d = 0; d < head_dim; ++d) {
                        int64_t k_idx = b * seq_len * num_heads * num_features + 
                                       s * num_heads * num_features + h * num_features + f;
                        int64_t v_idx = b * seq_len * num_heads * head_dim + 
                                       s * num_heads * head_dim + h * head_dim + d;
                        running_kv[f * head_dim + d] += k_features[k_idx] * v_ptr[v_idx];
                    }
                }
            }
        }
    }
}

size_t PerformerAttention::estimate_memory_usage(int64_t seq_len, int64_t head_dim) const {
    return seq_len * config_.num_features * sizeof(float) * 2 + // Q and K features
           config_.num_features * head_dim * sizeof(float);      // K^T * V matrix
}

void PerformerAttention::refresh_random_features() {
    random_features_.clear(); // Will be regenerated on next forward pass
}

// LinformerAttention Implementation
LinformerAttention::LinformerAttention(const Config& config) : LinearAttentionBase(config) {
    if (config_.scale < 0) {
        config_.scale = 1.0f / std::sqrt(64.0f); // Default head_dim
    }
}

Tensor LinformerAttention::forward(const Tensor& q, const Tensor& k, const Tensor& v) {
    const auto& q_shape = q.shape();
    int64_t batch_size = q_shape[0];
    int64_t seq_len = q_shape[1];
    int64_t num_heads = q_shape[2];
    int64_t head_dim = q_shape[3];
    
    // Initialize projections if needed
    if (!proj_k_ || !proj_v_) {
        initialize_projections(seq_len);
    }
    
    Tensor output(q_shape, q.dtype());
    
    linformer_attention_kernel(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        proj_k_->data_ptr<float>(), proj_v_->data_ptr<float>(),
        output.data_ptr<float>(), batch_size, seq_len, num_heads, head_dim, config_.projection_dim
    );
    
    return output;
}

void LinformerAttention::initialize_projections(int64_t seq_len) {
    // Create random projection matrices
    std::vector<int64_t> proj_shape = {seq_len, config_.projection_dim};
    proj_k_ = std::make_unique<Tensor>(proj_shape, core::DataType::FLOAT32);
    proj_v_ = std::make_unique<Tensor>(proj_shape, core::DataType::FLOAT32);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(seq_len)));
    
    float* proj_k_data = proj_k_->data_ptr<float>();
    float* proj_v_data = proj_v_->data_ptr<float>();
    
    for (int64_t i = 0; i < seq_len * config_.projection_dim; ++i) {
        proj_k_data[i] = dist(gen);
        proj_v_data[i] = dist(gen);
    }
}

void LinformerAttention::linformer_attention_kernel(
    const float* q_ptr, const float* k_ptr, const float* v_ptr,
    const float* proj_k_ptr, const float* proj_v_ptr,
    float* output_ptr, int64_t batch_size, int64_t seq_len,
    int64_t num_heads, int64_t head_dim, int64_t proj_dim) {
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            
            // Project K and V: K_proj = K * P_k, V_proj = V * P_v
            std::vector<float> k_proj(proj_dim * head_dim, 0.0f);
            std::vector<float> v_proj(proj_dim * head_dim, 0.0f);
            
            for (int64_t p = 0; p < proj_dim; ++p) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    float k_sum = 0.0f, v_sum = 0.0f;
                    for (int64_t s = 0; s < seq_len; ++s) {
                        int64_t k_idx = b * seq_len * num_heads * head_dim + 
                                       s * num_heads * head_dim + h * head_dim + d;
                        int64_t v_idx = k_idx;
                        int64_t proj_idx = s * proj_dim + p;
                        
                        k_sum += k_ptr[k_idx] * proj_k_ptr[proj_idx];
                        v_sum += v_ptr[v_idx] * proj_v_ptr[proj_idx];
                    }
                    k_proj[p * head_dim + d] = k_sum;
                    v_proj[p * head_dim + d] = v_sum;
                }
            }
            
            // Compute attention: Q * K_proj^T * V_proj
            for (int64_t i = 0; i < seq_len; ++i) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    
                    for (int64_t p = 0; p < proj_dim; ++p) {
                        // Q[i] * K_proj[p]
                        float qk = 0.0f;
                        for (int64_t d2 = 0; d2 < head_dim; ++d2) {
                            int64_t q_idx = b * seq_len * num_heads * head_dim + 
                                           i * num_heads * head_dim + h * head_dim + d2;
                            qk += q_ptr[q_idx] * k_proj[p * head_dim + d2];
                        }
                        
                        // Apply softmax (simplified - just scale)
                        qk *= config_.scale;
                        
                        // Multiply by V_proj[p]
                        sum += qk * v_proj[p * head_dim + d];
                    }
                    
                    int64_t out_idx = b * seq_len * num_heads * head_dim + 
                                     i * num_heads * head_dim + h * head_dim + d;
                    output_ptr[out_idx] = sum;
                }
            }
        }
    }
}

size_t LinformerAttention::estimate_memory_usage(int64_t seq_len, int64_t head_dim) const {
    return config_.projection_dim * head_dim * sizeof(float) * 2 + // K_proj, V_proj
           seq_len * config_.projection_dim * sizeof(float) * 2;    // Projection matrices
}

void LinformerAttention::set_key_projection(const Tensor& proj_k) {
    proj_k_ = std::make_unique<Tensor>(proj_k);
}

void LinformerAttention::set_value_projection(const Tensor& proj_v) {
    proj_v_ = std::make_unique<Tensor>(proj_v);
}

// CosformerAttention Implementation
CosformerAttention::CosformerAttention(const Config& config) : LinearAttentionBase(config) {
    if (config_.scale < 0) {
        config_.scale = 1.0f / std::sqrt(64.0f); // Default head_dim
    }
}

Tensor CosformerAttention::forward(const Tensor& q, const Tensor& k, const Tensor& v) {
    const auto& q_shape = q.shape();
    int64_t batch_size = q_shape[0];
    int64_t seq_len = q_shape[1];
    int64_t num_heads = q_shape[2];
    int64_t head_dim = q_shape[3];
    
    Tensor output(q_shape, q.dtype());
    
    cosformer_kernel(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, seq_len, num_heads, head_dim
    );
    
    return output;
}

void CosformerAttention::cosformer_kernel(
    const float* q_ptr, const float* k_ptr, const float* v_ptr,
    float* output_ptr, int64_t batch_size, int64_t seq_len,
    int64_t num_heads, int64_t head_dim) {
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            
            // Apply cosine ReLU to Q and K
            std::vector<float> q_cos(seq_len * head_dim);
            std::vector<float> k_cos(seq_len * head_dim);
            
            for (int64_t s = 0; s < seq_len; ++s) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    int64_t idx = b * seq_len * num_heads * head_dim + 
                                 s * num_heads * head_dim + h * head_dim + d;
                    int64_t local_idx = s * head_dim + d;
                    
                    q_cos[local_idx] = cosine_relu(q_ptr[idx]);
                    k_cos[local_idx] = cosine_relu(k_ptr[idx]);
                }
            }
            
            // Linear attention with cosine features
            for (int64_t i = 0; i < seq_len; ++i) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    
                    for (int64_t j = 0; j < seq_len; ++j) {
                        // Compute attention weight
                        float weight = 0.0f;
                        for (int64_t d2 = 0; d2 < head_dim; ++d2) {
                            weight += q_cos[i * head_dim + d2] * k_cos[j * head_dim + d2];
                        }
                        weight *= config_.scale;
                        
                        // Apply to value
                        int64_t v_idx = b * seq_len * num_heads * head_dim + 
                                       j * num_heads * head_dim + h * head_dim + d;
                        sum += weight * v_ptr[v_idx];
                    }
                    
                    int64_t out_idx = b * seq_len * num_heads * head_dim + 
                                     i * num_heads * head_dim + h * head_dim + d;
                    output_ptr[out_idx] = sum;
                }
            }
        }
    }
}

void CosformerAttention::cosine_relu_simd(const float* input, float* output, int64_t size) const {
    for (int64_t i = 0; i < size; ++i) {
        output[i] = cosine_relu(input[i]);
    }
}

size_t CosformerAttention::estimate_memory_usage(int64_t seq_len, int64_t head_dim) const {
    return seq_len * head_dim * sizeof(float) * 2; // Q_cos and K_cos
}

// LinearAttentionFactory Implementation
std::unique_ptr<LinearAttentionBase> LinearAttentionFactory::create(
    const LinearAttentionBase::Config& config) {
    switch (config.type) {
        case LinearAttentionType::PERFORMER:
            return std::make_unique<PerformerAttention>(config);
        case LinearAttentionType::LINFORMER:
            return std::make_unique<LinformerAttention>(config);
        case LinearAttentionType::COSFORMER:
            return std::make_unique<CosformerAttention>(config);
        default:
            return std::make_unique<PerformerAttention>(config);
    }
}

void LinearAttentionFactory::benchmark_methods(int64_t seq_len, int64_t head_dim, int64_t num_heads) {
    // Real benchmarking implementation would go here
}

float LinearAttentionFactory::compute_approximation_error(
    const Tensor& linear_output, const Tensor& exact_output) {
    // Compute MSE between linear and exact attention
    float mse = 0.0f;
    int64_t total_elements = linear_output.numel();
    
    const float* linear_data = linear_output.data_ptr<float>();
    const float* exact_data = exact_output.data_ptr<float>();
    
    for (int64_t i = 0; i < total_elements; ++i) {
        float diff = linear_data[i] - exact_data[i];
        mse += diff * diff;
    }
    
    return mse / total_elements;
}

void LinearAttentionFactory::compare_memory_usage(int64_t seq_len, int64_t head_dim) {
    // Real memory comparison implementation would go here
}

} // namespace attention
} // namespace operators
} // namespace deepcpp
