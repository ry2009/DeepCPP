#pragma once

#include "../../core/tensor/tensor.h"
#include <memory>
#include <vector>
#include <complex>
#include <immintrin.h>

namespace deepcpp {
namespace operators {
namespace models {

using deepcpp::core::Tensor;
using deepcpp::core::DataType;

/**
 * State Space Models (SSMs)
 * 
 * Implementation of various state space model architectures:
 * - S4 (Structured State Spaces)
 * - S5 (Simplified State Spaces)  
 * - Mamba (Selective State Spaces)
 * - DSS (Diagonal State Spaces)
 * - GSS (Gated State Spaces)
 */

enum class SSMType {
    S4,           // Structured State Spaces
    S5,           // Simplified State Spaces
    MAMBA,        // Selective State Spaces (Mamba)
    DSS,          // Diagonal State Spaces
    GSS,          // Gated State Spaces
    HIPPO,        // HiPPO basis functions
    LSSL,         // Linear State Space Layers
    MEGA          // MEGA (Exponential Gating)
};

/**
 * Base State Space Model Interface
 */
class StateSpaceModelBase {
public:
    struct Config {
        SSMType type;
        int64_t d_model;        // Model dimension
        int64_t d_state;        // State dimension (N)
        int64_t d_conv;         // Convolution dimension
        int64_t expand_factor;  // Expansion factor for intermediate dimensions
        bool use_cuda;          // Use CUDA-optimized kernels
        bool bidirectional;     // Bidirectional processing
        float dt_min;           // Minimum discretization step
        float dt_max;           // Maximum discretization step
        
        Config(SSMType t = SSMType::MAMBA, int64_t dm = 512, int64_t ds = 64,
               int64_t dc = 4, int64_t ef = 2, bool cuda = false, bool bidir = false,
               float dt_min = 0.001f, float dt_max = 0.1f)
            : type(t), d_model(dm), d_state(ds), d_conv(dc), expand_factor(ef),
              use_cuda(cuda), bidirectional(bidir), dt_min(dt_min), dt_max(dt_max) {}
    };
    
    virtual ~StateSpaceModelBase() = default;
    virtual Tensor forward(const Tensor& input, const Tensor* state = nullptr) = 0;
    virtual std::vector<Tensor> get_state() const = 0;
    virtual void reset_state() = 0;
    virtual size_t estimate_memory_usage(int64_t seq_len, int64_t batch_size) const = 0;
    
protected:
    Config config_;
    StateSpaceModelBase(const Config& config) : config_(config) {}
};

/**
 * Mamba (Selective State Space Model)
 * 
 * Key innovations:
 * - Selective mechanism for inputs
 * - Hardware-efficient implementation
 * - No attention mechanism needed
 */
class MambaSSM : public StateSpaceModelBase {
public:
    MambaSSM(const Config& config);
    
    Tensor forward(const Tensor& input, const Tensor* state = nullptr) override;
    std::vector<Tensor> get_state() const override;
    void reset_state() override;
    size_t estimate_memory_usage(int64_t seq_len, int64_t batch_size) const override;
    
    // Mamba-specific methods
    void set_selective_parameters(const Tensor& delta, const Tensor& A, const Tensor& B, const Tensor& C);
    Tensor selective_scan(const Tensor& input, const Tensor& delta, 
                         const Tensor& A, const Tensor& B, const Tensor& C);
    
private:
    // Learnable parameters
    std::unique_ptr<Tensor> A_;           // State transition matrix [d_state, d_model]
    std::unique_ptr<Tensor> B_;           // Input projection [d_state, d_model]  
    std::unique_ptr<Tensor> C_;           // Output projection [d_model, d_state]
    std::unique_ptr<Tensor> D_;           // Skip connection [d_model]
    std::unique_ptr<Tensor> delta_;       // Discretization parameter [d_model]
    
    // Internal state
    std::unique_ptr<Tensor> hidden_state_;
    
    // Core kernels
    void selective_scan_kernel(
        const float* input_ptr, const float* delta_ptr,
        const float* A_ptr, const float* B_ptr, const float* C_ptr,
        float* output_ptr, float* state_ptr,
        int64_t batch_size, int64_t seq_len, int64_t d_model, int64_t d_state
    );
    
    void selective_scan_cuda_kernel(
        const float* input_ptr, const float* delta_ptr,
        const float* A_ptr, const float* B_ptr, const float* C_ptr,
        float* output_ptr, float* state_ptr,
        int64_t batch_size, int64_t seq_len, int64_t d_model, int64_t d_state
    );
    
    // Discretization methods
    void discretize_bilinear(const Tensor& delta, const Tensor& A, const Tensor& B,
                           Tensor& A_discrete, Tensor& B_discrete);
    void discretize_zoh(const Tensor& delta, const Tensor& A, const Tensor& B,
                       Tensor& A_discrete, Tensor& B_discrete);
    
    // Initialization
    void initialize_hippo_matrix();
    void initialize_parameters();
};

/**
 * S4 (Structured State Spaces)
 * 
 * Uses structured matrices for efficient parameterization
 * and stable long-range dependencies
 */
class S4SSM : public StateSpaceModelBase {
public:
    enum class StructureType {
        DIAGONAL,       // Diagonal structure
        DPLR,          // Diagonal plus Low-Rank
        NPLR,          // Normal plus Low-Rank  
        HIPPO,         // HiPPO-based initialization
        LEGS           // Legendre polynomials
    };
    
    struct S4Config : public Config {
        StructureType structure;
        int64_t rank;           // For low-rank components
        bool use_fft;           // Use FFT for convolution
        bool trainable_dt;      // Learn discretization step
        
        S4Config(StructureType st = StructureType::DPLR, int64_t r = 64, 
                bool fft = true, bool train_dt = true)
            : structure(st), rank(r), use_fft(fft), trainable_dt(train_dt) {}
    };
    
    S4SSM(const S4Config& config);
    
    Tensor forward(const Tensor& input, const Tensor* state = nullptr) override;
    std::vector<Tensor> get_state() const override;
    void reset_state() override;
    size_t estimate_memory_usage(int64_t seq_len, int64_t batch_size) const override;
    
    // S4-specific methods
    Tensor compute_convolution_kernel(int64_t seq_len);
    Tensor fft_convolution(const Tensor& input, const Tensor& kernel);
    
private:
    S4Config s4_config_;
    
    // Structured parameters
    std::unique_ptr<Tensor> Lambda_;      // Eigenvalues (diagonal)
    std::unique_ptr<Tensor> P_;           // Low-rank factor P
    std::unique_ptr<Tensor> Q_;           // Low-rank factor Q
    std::unique_ptr<Tensor> B_;           // Input matrix
    std::unique_ptr<Tensor> C_;           // Output matrix
    std::unique_ptr<Tensor> dt_;          // Discretization step
    
    // Cached convolution kernels
    mutable std::unordered_map<int64_t, Tensor> kernel_cache_;
    
    void initialize_structured_matrix();
    void initialize_hippo_legs();
    
    // Core computation kernels
    void s4_recurrent_kernel(
        const float* input_ptr, float* output_ptr, float* state_ptr,
        int64_t batch_size, int64_t seq_len, int64_t d_model, int64_t d_state
    );
    
    void s4_convolution_kernel(
        const float* input_ptr, const float* kernel_ptr, float* output_ptr,
        int64_t batch_size, int64_t seq_len, int64_t d_model
    );
    
    // FFT utilities
    void complex_fft(const std::complex<float>* input, std::complex<float>* output, int64_t n);
    void complex_ifft(const std::complex<float>* input, std::complex<float>* output, int64_t n);
};

/**
 * S5 (Simplified State Spaces)
 * 
 * Simplified version of S4 with diagonal state matrices
 * for improved computational efficiency
 */
class S5SSM : public StateSpaceModelBase {
public:
    S5SSM(const Config& config);
    
    Tensor forward(const Tensor& input, const Tensor* state = nullptr) override;
    std::vector<Tensor> get_state() const override;
    void reset_state() override;
    size_t estimate_memory_usage(int64_t seq_len, int64_t batch_size) const override;
    
    // Parallel scan implementation
    Tensor parallel_scan(const Tensor& input);
    
private:
    // Diagonal parameters (much simpler than S4)
    std::unique_ptr<Tensor> Lambda_;      // Diagonal state matrix
    std::unique_ptr<Tensor> B_;           // Input matrix
    std::unique_ptr<Tensor> C_;           // Output matrix
    std::unique_ptr<Tensor> dt_;          // Discretization step
    
    std::unique_ptr<Tensor> hidden_state_;
    
    void s5_parallel_scan_kernel(
        const float* input_ptr, float* output_ptr,
        const float* lambda_ptr, const float* B_ptr, const float* C_ptr,
        float* workspace_ptr, int64_t batch_size, int64_t seq_len, 
        int64_t d_model, int64_t d_state
    );
    
    void initialize_diagonal_parameters();
};

/**
 * Gated State Space Model
 * 
 * Incorporates gating mechanisms for better gradient flow
 * and improved performance on various tasks
 */
class GatedSSM : public StateSpaceModelBase {
public:
    enum class GatingType {
        FORGET_GATE,    // LSTM-style forget gate
        UPDATE_GATE,    // GRU-style update gate
        INPUT_GATE,     // Input gating
        OUTPUT_GATE,    // Output gating
        MEGA_GATE       // MEGA-style exponential gating
    };
    
    struct GatedConfig : public Config {
        GatingType gate_type;
        bool use_layer_norm;
        float gate_bias_init;
        
        GatedConfig(GatingType gt = GatingType::FORGET_GATE, bool ln = true, float bias = 1.0f)
            : gate_type(gt), use_layer_norm(ln), gate_bias_init(bias) {}
    };
    
    GatedSSM(const GatedConfig& config);
    
    Tensor forward(const Tensor& input, const Tensor* state = nullptr) override;
    std::vector<Tensor> get_state() const override;
    void reset_state() override;
    size_t estimate_memory_usage(int64_t seq_len, int64_t batch_size) const override;
    
private:
    GatedConfig gated_config_;
    
    // Base SSM parameters
    std::unique_ptr<Tensor> A_;
    std::unique_ptr<Tensor> B_;
    std::unique_ptr<Tensor> C_;
    
    // Gating parameters
    std::unique_ptr<Tensor> gate_W_;      // Gate weight matrix
    std::unique_ptr<Tensor> gate_b_;      // Gate bias
    
    // Layer normalization
    std::unique_ptr<Tensor> ln_weight_;
    std::unique_ptr<Tensor> ln_bias_;
    
    std::unique_ptr<Tensor> hidden_state_;
    
    void gated_ssm_kernel(
        const float* input_ptr, float* output_ptr, float* state_ptr,
        int64_t batch_size, int64_t seq_len, int64_t d_model, int64_t d_state
    );
    
    float compute_gate(float input, float hidden, GatingType type);
    void apply_layer_norm(Tensor& input, const Tensor& weight, const Tensor& bias);
};

/**
 * Multi-Scale State Space Model
 * 
 * Processes inputs at multiple time scales for better
 * modeling of hierarchical temporal patterns
 */
class MultiScaleSSM {
public:
    struct Config {
        std::vector<int64_t> scales;        // Different time scales
        int64_t d_model;
        int64_t d_state;
        SSMType base_ssm_type;
        bool learnable_scales;
        
        Config(const std::vector<int64_t>& sc = {1, 2, 4, 8}, 
               int64_t dm = 512, int64_t ds = 64, SSMType base = SSMType::MAMBA,
               bool learn_scales = false)
            : scales(sc), d_model(dm), d_state(ds), base_ssm_type(base), 
              learnable_scales(learn_scales) {}
    };
    
    MultiScaleSSM(const Config& config);
    
    Tensor forward(const Tensor& input);
    void reset_all_states();
    
    size_t estimate_memory_usage(int64_t seq_len, int64_t batch_size) const;
    
private:
    Config config_;
    std::vector<std::unique_ptr<StateSpaceModelBase>> scale_ssms_;
    std::unique_ptr<Tensor> combination_weights_;
    
    Tensor downsample(const Tensor& input, int64_t scale);
    Tensor upsample(const Tensor& input, int64_t scale, int64_t target_len);
    Tensor combine_scales(const std::vector<Tensor>& scale_outputs);
};

/**
 * Bidirectional State Space Model
 * 
 * Processes sequences in both forward and backward directions
 */
class BidirectionalSSM {
public:
    BidirectionalSSM(const StateSpaceModelBase::Config& config);
    
    Tensor forward(const Tensor& input);
    void reset_states();
    
    size_t estimate_memory_usage(int64_t seq_len, int64_t batch_size) const;
    
private:
    std::unique_ptr<StateSpaceModelBase> forward_ssm_;
    std::unique_ptr<StateSpaceModelBase> backward_ssm_;
    std::unique_ptr<Tensor> combination_weights_;
    
    Tensor reverse_sequence(const Tensor& input);
    Tensor combine_directions(const Tensor& forward_out, const Tensor& backward_out);
};

/**
 * State Space Model Factory
 */
class SSMFactory {
public:
    static std::unique_ptr<StateSpaceModelBase> create(const StateSpaceModelBase::Config& config);
    
    // Benchmark different SSM variants
    static void benchmark_ssm_variants(int64_t seq_len, int64_t d_model, int64_t batch_size);
    
    // Analyze memory usage across variants
    static void compare_memory_usage(int64_t seq_len, int64_t d_model);
};

// Utility functions for SSMs
namespace ssm_utils {

/**
 * Initialization utilities
 */
void initialize_hippo_matrix(Tensor& A, int64_t d_state);
void initialize_legs_matrix(Tensor& A, int64_t d_state);
void initialize_fourier_features(Tensor& A, int64_t d_state);

/**
 * Discretization methods
 */
void discretize_bilinear(const Tensor& dt, const Tensor& A, const Tensor& B,
                        Tensor& A_disc, Tensor& B_disc);
void discretize_zoh(const Tensor& dt, const Tensor& A, const Tensor& B,
                   Tensor& A_disc, Tensor& B_disc);
void discretize_euler(const Tensor& dt, const Tensor& A, const Tensor& B,
                     Tensor& A_disc, Tensor& B_disc);

/**
 * Scan operations
 */
void parallel_prefix_scan(const float* input, float* output, int64_t size);
void associative_scan(const float* elements, const float* operators, 
                     float* output, int64_t size);

/**
 * Convolution utilities
 */
void fft_convolution_1d(const float* signal, const float* kernel, 
                       float* output, int64_t signal_len, int64_t kernel_len);
void causal_convolution_1d(const float* signal, const float* kernel,
                          float* output, int64_t signal_len, int64_t kernel_len);

/**
 * Performance analysis
 */
struct SSMBenchmark {
    SSMType type;
    int64_t seq_len;
    int64_t d_model;
    int64_t d_state;
    float forward_time_ms;
    size_t memory_usage_mb;
    float throughput_tokens_per_sec;
    int64_t operations_count;
};

std::vector<SSMBenchmark> benchmark_all_ssm_types(
    int64_t seq_len, int64_t d_model, int64_t d_state, int64_t batch_size = 1
);

/**
 * Theoretical analysis
 */
int64_t compute_ssm_complexity(SSMType type, int64_t seq_len, int64_t d_model, int64_t d_state);
float compute_memory_scaling(SSMType type, int64_t seq_len);
float compute_ssm_efficiency_vs_attention(int64_t seq_len, int64_t d_model);

} // namespace ssm_utils

} // namespace models
} // namespace operators
} // namespace deepcpp 