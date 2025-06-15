#pragma once

#include "../../core/tensor/tensor.h"
#include <vector>
#include <memory>
#include <random>
#include <immintrin.h>

namespace deepcpp {
namespace operators {
namespace attention {

using deepcpp::core::Tensor;
using deepcpp::core::DataType;

/**
 * Linear Attention Mechanisms
 * 
 * Collection of attention mechanisms that achieve O(n) complexity
 * instead of O(n²) through various approximation techniques.
 */

enum class LinearAttentionType {
    PERFORMER,       // FAVOR+ kernel approximation
    LINFORMER,       // Low-rank key/value projections
    SYNTHESIZER,     // Learned attention patterns
    COSFORMER,       // Cosine-based kernel
    RANDOM_FEATURES, // Random Fourier features
    ELU_KERNEL,      // ELU-based kernel
    RELU_KERNEL      // ReLU-based kernel
};

/**
 * Base Linear Attention Interface
 */
class LinearAttentionBase {
public:
    struct Config {
        LinearAttentionType type;
        int64_t projection_dim;    // For dimensionality reduction methods
        int64_t num_features;      // For kernel methods
        float scale;
        bool causal;
        
        Config(LinearAttentionType t = LinearAttentionType::PERFORMER,
               int64_t proj_dim = 256, int64_t nf = 256, float s = -1.0f, bool c = false)
            : type(t), projection_dim(proj_dim), num_features(nf), scale(s), causal(c) {}
    };
    
    virtual ~LinearAttentionBase() = default;
    virtual Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v) = 0;
    virtual size_t estimate_memory_usage(int64_t seq_len, int64_t head_dim) const = 0;
    
protected:
    Config config_;
    LinearAttentionBase(const Config& config) : config_(config) {}
};

/**
 * Performer Attention (FAVOR+)
 * 
 * Uses random feature maps to approximate the softmax kernel
 * with unbiased estimation and linear complexity.
 */
class PerformerAttention : public LinearAttentionBase {
public:
    PerformerAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v) override;
    size_t estimate_memory_usage(int64_t seq_len, int64_t head_dim) const override;
    
    // Update random features (useful for training)
    void refresh_random_features();
    
private:
    std::vector<std::vector<float>> random_features_;  // [num_heads][num_features * head_dim]
    std::mt19937 rng_;
    
    void initialize_random_features(int64_t num_heads, int64_t head_dim);
    
    // FAVOR+ kernel approximation
    void compute_feature_maps(
        const float* input_ptr, float* feature_ptr,
        int64_t batch_size, int64_t seq_len, int64_t num_heads, int64_t head_dim,
        const std::vector<std::vector<float>>& features, bool is_query
    );
    
    void linear_attention_kernel(
        const float* q_features, const float* k_features, const float* v_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len,
        int64_t num_heads, int64_t head_dim, int64_t num_features
    );
    
    void causal_linear_attention_kernel(
        const float* q_features, const float* k_features, const float* v_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len,
        int64_t num_heads, int64_t head_dim, int64_t num_features
    );
};

/**
 * Linformer Attention
 * 
 * Projects keys and values to lower dimension using learned projections
 */
class LinformerAttention : public LinearAttentionBase {
public:
    LinformerAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v) override;
    size_t estimate_memory_usage(int64_t seq_len, int64_t head_dim) const override;
    
    // Set projection matrices (learned parameters)
    void set_key_projection(const Tensor& proj_k);
    void set_value_projection(const Tensor& proj_v);
    
private:
    std::unique_ptr<Tensor> proj_k_;  // [seq_len, projection_dim]
    std::unique_ptr<Tensor> proj_v_;  // [seq_len, projection_dim]
    
    void initialize_projections(int64_t seq_len);
    
    void linformer_attention_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        const float* proj_k_ptr, const float* proj_v_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len,
        int64_t num_heads, int64_t head_dim, int64_t proj_dim
    );
};

/**
 * Cosformer Attention
 * 
 * Uses cosine-based ReLU kernel for linear attention
 */
class CosformerAttention : public LinearAttentionBase {
public:
    CosformerAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v) override;
    size_t estimate_memory_usage(int64_t seq_len, int64_t head_dim) const override;
    
private:
    void cosformer_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len,
        int64_t num_heads, int64_t head_dim
    );
    
    // Cosine ReLU activation: cos(x) * ReLU(x)
    inline float cosine_relu(float x) const {
        return x > 0 ? std::cos(x) * x : 0.0f;
    }
    
    // Optimized SIMD version
    void cosine_relu_simd(const float* input, float* output, int64_t size) const;
};

/**
 * Synthesizer Attention
 * 
 * Uses learned synthetic attention patterns instead of content-based attention
 */
class SynthesizerAttention : public LinearAttentionBase {
public:
    enum class SynthesizerType {
        DENSE,      // Dense learned attention
        RANDOM,     // Random fixed attention
        MIXED       // Combination of dense and random
    };
    
    struct SynthesizerConfig : public Config {
        SynthesizerType synth_type;
        float dense_ratio;  // For mixed type
        
        SynthesizerConfig(SynthesizerType st = SynthesizerType::DENSE, float dr = 0.5f)
            : synth_type(st), dense_ratio(dr) {}
    };
    
    SynthesizerAttention(const SynthesizerConfig& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v) override;
    size_t estimate_memory_usage(int64_t seq_len, int64_t head_dim) const override;
    
    // Set learned attention patterns
    void set_attention_weights(const Tensor& weights);
    
private:
    SynthesizerConfig synth_config_;
    std::unique_ptr<Tensor> attention_weights_;  // [num_heads, seq_len, seq_len]
    std::unique_ptr<Tensor> random_weights_;     // Fixed random patterns
    
    void initialize_synthetic_patterns(int64_t num_heads, int64_t seq_len);
    
    void synthesizer_kernel(
        const float* q_ptr, const float* v_ptr, const float* weights_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len,
        int64_t num_heads, int64_t head_dim
    );
};

/**
 * Random Features Attention
 * 
 * Uses random Fourier features to approximate attention kernels
 */
class RandomFeaturesAttention : public LinearAttentionBase {
public:
    enum class KernelType {
        RBF,        // Radial basis function
        POLYNOMIAL, // Polynomial kernel
        LAPLACE,    // Laplace kernel
        MATERN      // Matérn kernel
    };
    
    struct RandomFeaturesConfig : public Config {
        KernelType kernel_type;
        float bandwidth;    // For RBF/Laplace kernels
        int degree;         // For polynomial kernels
        float nu;           // For Matérn kernels
        
        RandomFeaturesConfig(KernelType kt = KernelType::RBF, float bw = 1.0f, 
                           int deg = 2, float n = 1.5f)
            : kernel_type(kt), bandwidth(bw), degree(deg), nu(n) {}
    };
    
    RandomFeaturesAttention(const RandomFeaturesConfig& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v) override;
    size_t estimate_memory_usage(int64_t seq_len, int64_t head_dim) const override;
    
private:
    RandomFeaturesConfig rf_config_;
    std::vector<std::vector<float>> feature_weights_;  // Random feature weights
    std::vector<float> feature_biases_;                // Random biases
    
    void initialize_random_features(int64_t head_dim);
    
    void compute_random_features(
        const float* input_ptr, float* features_ptr,
        int64_t batch_size, int64_t seq_len, int64_t head_dim,
        const std::vector<float>& weights, const std::vector<float>& biases
    );
    
    void random_features_kernel(
        const float* q_features, const float* k_features, const float* v_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len,
        int64_t num_heads, int64_t head_dim, int64_t num_features
    );
};

/**
 * ELU-based Linear Attention
 * 
 * Uses ELU activation as a kernel function for linear attention
 */
class ELULinearAttention : public LinearAttentionBase {
public:
    ELULinearAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v) override;
    size_t estimate_memory_usage(int64_t seq_len, int64_t head_dim) const override;
    
private:
    float alpha_;  // ELU parameter
    
    void elu_linear_attention_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len,
        int64_t num_heads, int64_t head_dim
    );
    
    // ELU activation: x if x > 0, alpha * (exp(x) - 1) if x <= 0
    inline float elu(float x, float alpha) const {
        return x > 0 ? x : alpha * (std::exp(x) - 1.0f);
    }
    
    // Vectorized ELU
    void elu_simd(const float* input, float* output, int64_t size, float alpha) const;
};

/**
 * Efficient Linear Attention Manager
 * 
 * Factory and utility class for linear attention mechanisms
 */
class LinearAttentionFactory {
public:
    static std::unique_ptr<LinearAttentionBase> create(
        const LinearAttentionBase::Config& config
    );
    
    // Benchmark different linear attention methods
    static void benchmark_methods(int64_t seq_len, int64_t head_dim, int64_t num_heads);
    
    // Analyze approximation quality
    static float compute_approximation_error(
        const Tensor& linear_output, const Tensor& exact_output
    );
    
    // Memory usage comparison
    static void compare_memory_usage(int64_t seq_len, int64_t head_dim);
};

/**
 * Adaptive Linear Attention
 * 
 * Dynamically switches between different linear attention methods
 * based on sequence length and available memory
 */
class AdaptiveLinearAttention {
public:
    struct AdaptiveConfig {
        int64_t short_seq_threshold;    // Use standard attention for short sequences
        int64_t medium_seq_threshold;   // Switch between methods
        LinearAttentionType short_method;
        LinearAttentionType medium_method;
        LinearAttentionType long_method;
        
        AdaptiveConfig(int64_t short_thresh = 512, int64_t med_thresh = 4096)
            : short_seq_threshold(short_thresh), medium_seq_threshold(med_thresh),
              short_method(LinearAttentionType::PERFORMER),
              medium_method(LinearAttentionType::LINFORMER),
              long_method(LinearAttentionType::COSFORMER) {}
    };
    
    AdaptiveLinearAttention(const AdaptiveConfig& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v);
    
    // Get current method being used
    LinearAttentionType get_current_method() const { return current_method_; }
    
private:
    AdaptiveConfig config_;
    LinearAttentionType current_method_;
    std::unordered_map<LinearAttentionType, std::unique_ptr<LinearAttentionBase>> attention_methods_;
    
    LinearAttentionType select_method(int64_t seq_len) const;
    void initialize_methods();
};

// Utility functions for linear attention
namespace linear_utils {

/**
 * Kernel approximation utilities
 */
float compute_kernel_approximation_error(
    const float* exact_kernel, const float* approx_kernel, int64_t size
);

void generate_random_features_rbf(
    std::vector<float>& weights, std::vector<float>& biases,
    int64_t input_dim, int64_t num_features, float bandwidth, int seed = 42
);

void generate_random_features_polynomial(
    std::vector<float>& weights, int64_t input_dim, int64_t num_features,
    int degree, int seed = 42
);

/**
 * Performance analysis
 */
struct LinearAttentionBenchmark {
    LinearAttentionType method;
    float forward_time_ms;
    float memory_usage_mb;
    float approximation_error;
    int64_t operations_count;
};

std::vector<LinearAttentionBenchmark> benchmark_all_methods(
    int64_t seq_len, int64_t head_dim, int64_t num_heads, int64_t batch_size = 1
);

/**
 * Theoretical analysis
 */
int64_t compute_complexity_standard_attention(int64_t seq_len, int64_t head_dim);
int64_t compute_complexity_linear_attention(int64_t seq_len, int64_t head_dim, int64_t num_features);
float compute_memory_reduction_ratio(int64_t seq_len, int64_t head_dim, int64_t num_features);

} // namespace linear_utils

} // namespace attention
} // namespace operators
} // namespace deepcpp 