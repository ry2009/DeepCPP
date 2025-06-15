#pragma once

#include "../../core/tensor/tensor.h"
#include <immintrin.h>
#include <memory>
#include <vector>

namespace deepcpp {
namespace operators {
namespace performance {

using deepcpp::core::Tensor;
using deepcpp::core::DataType;

/**
 * SIMD Optimized Kernels for Deep Learning Operations
 * 
 * Provides highly optimized SIMD implementations for:
 * - Matrix multiplication (GEMM)
 * - Attention computations
 * - Activation functions
 * - Normalization layers
 * - Convolution operations
 */

/**
 * SIMD Configuration and Detection
 */
struct SIMDConfig {
    bool has_avx2;
    bool has_avx512f;
    bool has_avx512bw;
    bool has_avx512vl;
    bool has_fma;
    int vector_width;           // Number of floats in SIMD register
    int preferred_alignment;    // Memory alignment for optimal performance
    
    SIMDConfig();
    static SIMDConfig detect();
    void print_capabilities() const;
};

/**
 * AVX2 Optimized Kernels
 */
namespace avx2 {

/**
 * Matrix Multiplication (GEMM) Kernels
 */
void gemm_f32(const float* A, const float* B, float* C,
              int64_t M, int64_t N, int64_t K,
              float alpha = 1.0f, float beta = 0.0f);

void gemm_f32_batched(const float* A, const float* B, float* C,
                      int64_t batch_size, int64_t M, int64_t N, int64_t K,
                      float alpha = 1.0f, float beta = 0.0f);

// Specialized GEMM for common sizes
void gemm_4x4_kernel(const float* A, const float* B, float* C, int64_t K);
void gemm_8x8_kernel(const float* A, const float* B, float* C, int64_t K);
void gemm_16x16_kernel(const float* A, const float* B, float* C, int64_t K);

/**
 * Attention Kernels
 */
void attention_qk_kernel(const float* Q, const float* K, float* scores,
                        int64_t batch_size, int64_t seq_len, int64_t head_dim,
                        float scale);

void attention_softmax_kernel(float* scores, int64_t batch_size, 
                             int64_t num_heads, int64_t seq_len);

void attention_weighted_sum_kernel(const float* scores, const float* V, float* output,
                                  int64_t batch_size, int64_t num_heads,
                                  int64_t seq_len, int64_t head_dim);

// Flash attention optimized kernels
void flash_attention_forward_kernel(const float* Q, const float* K, const float* V,
                                   float* output, float* workspace,
                                   int64_t batch_size, int64_t seq_len,
                                   int64_t num_heads, int64_t head_dim,
                                   int64_t block_size, float scale);

/**
 * Activation Function Kernels
 */
void gelu_kernel(const float* input, float* output, int64_t size);
void gelu_backward_kernel(const float* input, const float* grad_output, 
                         float* grad_input, int64_t size);

void swish_kernel(const float* input, float* output, int64_t size, float beta = 1.0f);
void swish_backward_kernel(const float* input, const float* grad_output,
                          float* grad_input, int64_t size, float beta = 1.0f);

void relu_kernel(const float* input, float* output, int64_t size);
void relu6_kernel(const float* input, float* output, int64_t size);
void leaky_relu_kernel(const float* input, float* output, int64_t size, float alpha = 0.01f);

void mish_kernel(const float* input, float* output, int64_t size);
void geglu_kernel(const float* input, float* output, int64_t size, int64_t hidden_dim);

/**
 * Normalization Kernels
 */
void layer_norm_kernel(const float* input, const float* weight, const float* bias,
                      float* output, float* mean, float* var,
                      int64_t batch_size, int64_t seq_len, int64_t hidden_size,
                      float epsilon = 1e-5f);

void rms_norm_kernel(const float* input, const float* weight,
                    float* output, float* rms,
                    int64_t batch_size, int64_t seq_len, int64_t hidden_size,
                    float epsilon = 1e-5f);

void group_norm_kernel(const float* input, const float* weight, const float* bias,
                      float* output, int64_t batch_size, int64_t channels,
                      int64_t height, int64_t width, int64_t num_groups,
                      float epsilon = 1e-5f);

/**
 * Element-wise Operations
 */
void add_kernel(const float* a, const float* b, float* c, int64_t size);
void sub_kernel(const float* a, const float* b, float* c, int64_t size);
void mul_kernel(const float* a, const float* b, float* c, int64_t size);
void div_kernel(const float* a, const float* b, float* c, int64_t size);

void add_scalar_kernel(const float* input, float scalar, float* output, int64_t size);
void scale_kernel(const float* input, float scale, float* output, int64_t size);

/**
 * Reduction Operations
 */
float sum_kernel(const float* input, int64_t size);
float mean_kernel(const float* input, int64_t size);
float max_kernel(const float* input, int64_t size);
float min_kernel(const float* input, int64_t size);

void sum_axis_kernel(const float* input, float* output, 
                    int64_t batch_size, int64_t seq_len, int64_t hidden_size,
                    int axis);

/**
 * Convolution Kernels
 */
void conv1d_kernel(const float* input, const float* weight, const float* bias,
                  float* output, int64_t batch_size, int64_t in_channels,
                  int64_t out_channels, int64_t seq_len, int64_t kernel_size,
                  int64_t stride = 1, int64_t padding = 0);

void depthwise_conv1d_kernel(const float* input, const float* weight, const float* bias,
                            float* output, int64_t batch_size, int64_t channels,
                            int64_t seq_len, int64_t kernel_size,
                            int64_t stride = 1, int64_t padding = 0);

} // namespace avx2

/**
 * AVX-512 Optimized Kernels
 */
namespace avx512 {

// Enhanced versions of AVX2 kernels with 512-bit registers
void gemm_f32(const float* A, const float* B, float* C,
              int64_t M, int64_t N, int64_t K,
              float alpha = 1.0f, float beta = 0.0f);

void attention_qk_kernel(const float* Q, const float* K, float* scores,
                        int64_t batch_size, int64_t seq_len, int64_t head_dim,
                        float scale);

void flash_attention_kernel(const float* Q, const float* K, const float* V,
                           float* output, float* workspace,
                           int64_t batch_size, int64_t seq_len,
                           int64_t num_heads, int64_t head_dim,
                           int64_t block_size, float scale);

// Specialized kernels for large batch processing
void batch_layer_norm_kernel(const float* input, const float* weight, const float* bias,
                            float* output, int64_t batch_size, int64_t seq_len,
                            int64_t hidden_size, float epsilon = 1e-5f);

void batch_attention_kernel(const float* Q, const float* K, const float* V,
                           float* output, int64_t batch_size, int64_t num_heads,
                           int64_t seq_len, int64_t head_dim, float scale);

} // namespace avx512

/**
 * Adaptive SIMD Kernel Dispatcher
 * 
 * Automatically selects the best available SIMD implementation
 */
class SIMDKernelDispatcher {
public:
    SIMDKernelDispatcher();
    
    // GEMM operations
    void gemm(const float* A, const float* B, float* C,
              int64_t M, int64_t N, int64_t K,
              float alpha = 1.0f, float beta = 0.0f) const;
    
    // Attention operations
    void attention_forward(const float* Q, const float* K, const float* V,
                          float* output, int64_t batch_size, int64_t num_heads,
                          int64_t seq_len, int64_t head_dim, float scale) const;
    
    // Activation functions
    void gelu(const float* input, float* output, int64_t size) const;
    void swish(const float* input, float* output, int64_t size, float beta = 1.0f) const;
    void relu(const float* input, float* output, int64_t size) const;
    
    // Normalization
    void layer_norm(const float* input, const float* weight, const float* bias,
                   float* output, int64_t batch_size, int64_t seq_len,
                   int64_t hidden_size, float epsilon = 1e-5f) const;
    
    void rms_norm(const float* input, const float* weight, float* output,
                 int64_t batch_size, int64_t seq_len, int64_t hidden_size,
                 float epsilon = 1e-5f) const;
    
    // Element-wise operations
    void add(const float* a, const float* b, float* c, int64_t size) const;
    void mul(const float* a, const float* b, float* c, int64_t size) const;
    void scale(const float* input, float scale_factor, float* output, int64_t size) const;
    
    // Configuration
    const SIMDConfig& get_config() const { return config_; }
    void set_num_threads(int num_threads) { num_threads_ = num_threads; }
    
private:
    SIMDConfig config_;
    int num_threads_;
    
    // Function pointers for dynamic dispatch
    void (*gemm_func_)(const float*, const float*, float*, int64_t, int64_t, int64_t, float, float);
    void (*gelu_func_)(const float*, float*, int64_t);
    void (*layer_norm_func_)(const float*, const float*, const float*, float*, int64_t, int64_t, int64_t, float);
    
    void initialize_function_pointers();
};

/**
 * SIMD Memory Utilities
 */
namespace simd_memory {

// Aligned memory allocation
void* aligned_alloc(size_t size, size_t alignment = 64);
void aligned_free(void* ptr);

// Memory copy optimized for SIMD
void memcpy_simd(void* dst, const void* src, size_t size);
void memset_simd(void* ptr, int value, size_t size);

// Prefetching utilities
void prefetch_read(const void* ptr);
void prefetch_write(void* ptr);

} // namespace simd_memory

/**
 * SIMD Tensor Operations
 * 
 * High-level tensor operations using SIMD kernels
 */
class SIMDTensorOps {
public:
    static SIMDTensorOps& instance();
    
    // Matrix operations
    Tensor matmul(const Tensor& a, const Tensor& b) const;
    Tensor addmm(const Tensor& bias, const Tensor& a, const Tensor& b, 
                 float alpha = 1.0f, float beta = 1.0f) const;
    
    // Attention operations
    Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v,
                    const Tensor* mask = nullptr, float scale = -1.0f) const;
    
    Tensor flash_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                          int64_t block_size = 64, float scale = -1.0f) const;
    
    // Activation functions
    Tensor gelu(const Tensor& input) const;
    Tensor swish(const Tensor& input, float beta = 1.0f) const;
    Tensor relu(const Tensor& input) const;
    Tensor mish(const Tensor& input) const;
    
    // Normalization
    Tensor layer_norm(const Tensor& input, const Tensor& weight, 
                     const Tensor& bias, float epsilon = 1e-5f) const;
    
    Tensor rms_norm(const Tensor& input, const Tensor& weight,
                   float epsilon = 1e-5f) const;
    
    // Element-wise operations
    Tensor add(const Tensor& a, const Tensor& b) const;
    Tensor sub(const Tensor& a, const Tensor& b) const;
    Tensor mul(const Tensor& a, const Tensor& b) const;
    Tensor div(const Tensor& a, const Tensor& b) const;
    
    // Reduction operations
    Tensor sum(const Tensor& input, int64_t dim = -1) const;
    Tensor mean(const Tensor& input, int64_t dim = -1) const;
    Tensor max(const Tensor& input, int64_t dim = -1) const;
    
    // Performance monitoring
    void enable_profiling(bool enable = true) { profiling_enabled_ = enable; }
    void print_performance_stats() const;
    void reset_performance_stats();
    
private:
    SIMDTensorOps();
    
    std::unique_ptr<SIMDKernelDispatcher> dispatcher_;
    mutable bool profiling_enabled_;
    mutable std::unordered_map<std::string, float> operation_times_;
    mutable std::unordered_map<std::string, int64_t> operation_counts_;
    
    void profile_operation(const std::string& op_name, float time_ms) const;
};

// Utility functions
namespace simd_utils {

/**
 * Performance testing utilities
 */
struct SIMDPerformanceTest {
    std::string operation;
    std::string implementation;
    float time_ms;
    float gflops;
    size_t memory_bandwidth_gb_s;
};

std::vector<SIMDPerformanceTest> benchmark_simd_kernels(
    const std::vector<int64_t>& sizes = {1024, 4096, 16384}
);

/**
 * Memory alignment utilities
 */
bool is_aligned(const void* ptr, size_t alignment);
size_t get_alignment_offset(const void* ptr, size_t alignment);
void* align_pointer(void* ptr, size_t alignment);

/**
 * SIMD debugging utilities
 */
void print_m256(__m256 vec, const char* name = "vec");
void print_m256i(__m256i vec, const char* name = "vec");
void print_m512(__m512 vec, const char* name = "vec");

/**
 * Compiler optimization hints
 */
#define SIMD_INLINE __forceinline
#define SIMD_NOINLINE __declspec(noinline)
#define SIMD_RESTRICT __restrict
#define SIMD_ASSUME_ALIGNED(ptr, alignment) __assume_aligned(ptr, alignment)

} // namespace simd_utils

} // namespace performance
} // namespace operators
} // namespace deepcpp 