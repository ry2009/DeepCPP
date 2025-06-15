#include "../custom_ops.h"
#include <vector>
#include <cmath>
#include <immintrin.h>  // For AVX2 intrinsics
#include <omp.h>        // For OpenMP
#include <algorithm>
#include <memory>

namespace ssm_kernels {

// High-performance selective SSM scan implementation
// Based on Mamba's selective scan with optimizations for CPU
void selective_scan_optimized(
    const float* x,      // Input tensor [B, L, D]
    const float* delta,  // Delta tensor [B, L, D] 
    const float* A,      // A matrix [D, N]
    const float* B,      // B tensor [B, L, N]
    const float* C,      // C tensor [B, L, N]
    float* y,            // Output tensor [B, L, D]
    int64_t batch_size,  // B
    int64_t seq_len,     // L
    int64_t d_model,     // D
    int64_t d_state      // N
) {
    // Allocate workspace for hidden states
    std::vector<float> h_workspace(batch_size * d_model * d_state, 0.0f);
    float* h = h_workspace.data();

    // Process each batch in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int64_t b = 0; b < batch_size; ++b) {
        // Batch offsets
        const float* x_b = x + b * seq_len * d_model;
        const float* delta_b = delta + b * seq_len * d_model;
        const float* B_b = B + b * seq_len * d_state;
        const float* C_b = C + b * seq_len * d_state;
        float* y_b = y + b * seq_len * d_model;
        float* h_b = h + b * d_model * d_state;

        // Initialize hidden state to zero
        std::fill(h_b, h_b + d_model * d_state, 0.0f);

        // Sequential scan over time steps
        for (int64_t t = 0; t < seq_len; ++t) {
            const float* x_t = x_b + t * d_model;
            const float* delta_t = delta_b + t * d_model;
            const float* B_t = B_b + t * d_state;
            const float* C_t = C_b + t * d_state;
            float* y_t = y_b + t * d_model;

            // Update hidden state and compute output for each dimension
            for (int64_t d = 0; d < d_model; ++d) {
                float* h_d = h_b + d * d_state;
                const float* A_d = A + d * d_state;
                
                float dt = delta_t[d];
                float input = x_t[d];
                
                // Discretize A: A_discrete = exp(dt * A)
                // Update hidden state: h = A_discrete * h + dt * B * input
                // Vectorized inner loop for state dimension
                float output = 0.0f;
                
                #pragma omp simd reduction(+:output)
                for (int64_t n = 0; n < d_state; ++n) {
                    float A_discrete = std::exp(dt * A_d[n]);
                    h_d[n] = A_discrete * h_d[n] + dt * B_t[n] * input;
                    output += C_t[n] * h_d[n];
                }
                
                y_t[d] = output;
            }
        }
    }
}

// AVX2-optimized version for larger state dimensions
void selective_scan_avx2(
    const float* x, const float* delta, const float* A, 
    const float* B, const float* C, float* y,
    int64_t batch_size, int64_t seq_len, int64_t d_model, int64_t d_state
) {
    const int64_t avx_width = 8; // 8 floats per AVX2 register
    
    if (d_state % avx_width != 0) {
        // Fallback to scalar version
        selective_scan_optimized(x, delta, A, B, C, y, batch_size, seq_len, d_model, d_state);
        return;
    }

    std::vector<float> h_workspace(batch_size * d_model * d_state, 0.0f);
    float* h = h_workspace.data();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* x_b = x + b * seq_len * d_model;
        const float* delta_b = delta + b * seq_len * d_model;
        const float* B_b = B + b * seq_len * d_state;
        const float* C_b = C + b * seq_len * d_state;
        float* y_b = y + b * seq_len * d_model;
        float* h_b = h + b * d_model * d_state;

        for (int64_t t = 0; t < seq_len; ++t) {
            const float* x_t = x_b + t * d_model;
            const float* delta_t = delta_b + t * d_model;
            const float* B_t = B_b + t * d_state;
            const float* C_t = C_b + t * d_state;
            float* y_t = y_b + t * d_model;

            for (int64_t d = 0; d < d_model; ++d) {
                float* h_d = h_b + d * d_state;
                const float* A_d = A + d * d_state;
                
                float dt = delta_t[d];
                float input = x_t[d];
                
                __m256 dt_vec = _mm256_set1_ps(dt);
                __m256 input_vec = _mm256_set1_ps(input);
                __m256 output_vec = _mm256_setzero_ps();
                
                for (int64_t n = 0; n < d_state; n += avx_width) {
                    __m256 A_vec = _mm256_loadu_ps(&A_d[n]);
                    __m256 h_vec = _mm256_loadu_ps(&h_d[n]);
                    __m256 B_vec = _mm256_loadu_ps(&B_t[n]);
                    __m256 C_vec = _mm256_loadu_ps(&C_t[n]);
                    
                    // A_discrete = exp(dt * A)
                    __m256 A_discrete = _mm256_mul_ps(dt_vec, A_vec);
                    // Note: _mm256_exp_ps not available, use scalar fallback for exp
                    alignas(32) float A_discrete_scalar[8];
                    _mm256_store_ps(A_discrete_scalar, A_discrete);
                    for (int i = 0; i < 8; ++i) {
                        A_discrete_scalar[i] = std::exp(A_discrete_scalar[i]);
                    }
                    A_discrete = _mm256_load_ps(A_discrete_scalar);
                    
                    // h = A_discrete * h + dt * B * input
                    __m256 new_h = _mm256_fmadd_ps(A_discrete, h_vec, 
                                     _mm256_mul_ps(dt_vec, _mm256_mul_ps(B_vec, input_vec)));
                    _mm256_storeu_ps(&h_d[n], new_h);
                    
                    // output += C * h
                    output_vec = _mm256_fmadd_ps(C_vec, new_h, output_vec);
                }
                
                // Horizontal sum of output_vec
                alignas(32) float output_scalar[8];
                _mm256_store_ps(output_scalar, output_vec);
                y_t[d] = output_scalar[0] + output_scalar[1] + output_scalar[2] + output_scalar[3] +
                         output_scalar[4] + output_scalar[5] + output_scalar[6] + output_scalar[7];
            }
        }
    }
}

// Memory-efficient block-wise processing for very large sequences
void selective_scan_blocked(
    const float* x, const float* delta, const float* A, 
    const float* B, const float* C, float* y,
    int64_t batch_size, int64_t seq_len, int64_t d_model, int64_t d_state,
    int64_t block_size = 512
) {
    std::vector<float> h_workspace(batch_size * d_model * d_state, 0.0f);
    float* h = h_workspace.data();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t b = 0; b < batch_size; ++b) {
        float* h_b = h + b * d_model * d_state;
        std::fill(h_b, h_b + d_model * d_state, 0.0f);

        // Process sequence in blocks
        for (int64_t block_start = 0; block_start < seq_len; block_start += block_size) {
            int64_t block_end = std::min(block_start + block_size, seq_len);
            
            selective_scan_optimized(
                x + b * seq_len * d_model + block_start * d_model,
                delta + b * seq_len * d_model + block_start * d_model,
                A,
                B + b * seq_len * d_state + block_start * d_state,
                C + b * seq_len * d_state + block_start * d_state,
                y + b * seq_len * d_model + block_start * d_model,
                1, // Single batch
                block_end - block_start,
                d_model,
                d_state
            );
        }
    }
}

// Auto-dispatch to best implementation based on problem size
void selective_scan_auto(
    const float* x, const float* delta, const float* A, 
    const float* B, const float* C, float* y,
    int64_t batch_size, int64_t seq_len, int64_t d_model, int64_t d_state
) {
    // Choose implementation based on problem characteristics
    int64_t total_ops = batch_size * seq_len * d_model * d_state;
    
    if (total_ops > 100000000 && seq_len > 2048) {
        // Very large problem: use blocked processing
        selective_scan_blocked(x, delta, A, B, C, y, batch_size, seq_len, d_model, d_state);
    } else if (d_state >= 64 && d_state % 8 == 0) {
        // Medium problem with good vectorization: use AVX2
        selective_scan_avx2(x, delta, A, B, C, y, batch_size, seq_len, d_model, d_state);
    } else {
        // Small/irregular problem: use optimized scalar
        selective_scan_optimized(x, delta, A, B, C, y, batch_size, seq_len, d_model, d_state);
    }
}

// Reference implementation of selective SSM scan (original)
void selective_scan_ref(
    const float* x, const float* delta, const float* A, 
    const float* B, const float* C, float* y,
    int64_t batch_size, int64_t seq_len, int64_t d_model, int64_t d_state
) {
    // Use the auto-dispatch for best performance
    selective_scan_auto(x, delta, A, B, C, y, batch_size, seq_len, d_model, d_state);
}

} // namespace ssm_kernels

// Actual kernel computation that will be called from custom_ops.cc
void ssm_scan_compute(
    const float* x, const float* delta, const float* A, 
    const float* B, const float* C, float* y,
    const std::vector<int64_t>& x_shape
) {
    // Parse input shapes
    // Assume x_shape is [B, L, D]
    int64_t batch_size = x_shape[0];
    int64_t seq_len = x_shape[1]; 
    int64_t d_model = x_shape[2];
    
    // Assume d_state = 16 for now (typical Mamba setting)
    int64_t d_state = 16;
    
    // Choose implementation based on capabilities
    bool use_avx2 = true;  // Could detect CPU features
    
    if (use_avx2) {
        ssm_kernels::selective_scan_avx2(x, delta, A, B, C, y, 
                                        batch_size, seq_len, d_model, d_state);
    } else {
        ssm_kernels::selective_scan_ref(x, delta, A, B, C, y,
                                       batch_size, seq_len, d_model, d_state);
    }
} 