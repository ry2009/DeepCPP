#include "ssm.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <complex>
#include <random>

namespace deepcpp {
namespace operators {
namespace models {

// MambaSSM Implementation
MambaSSM::MambaSSM(const Config& config) : StateSpaceModelBase(config) {
    initialize_parameters();
}

void MambaSSM::initialize_parameters() {
    int64_t d_model = config_.d_model;
    int64_t d_state = config_.d_state;
    
    // Initialize state transition matrix A (diagonal structure)
    std::vector<int64_t> A_shape = {d_state, d_model};
    A_ = std::make_unique<Tensor>(A_shape, core::DataType::FLOAT32);
    
    // Initialize input projection B
    std::vector<int64_t> B_shape = {d_state, d_model};
    B_ = std::make_unique<Tensor>(B_shape, core::DataType::FLOAT32);
    
    // Initialize output projection C
    std::vector<int64_t> C_shape = {d_model, d_state};
    C_ = std::make_unique<Tensor>(C_shape, core::DataType::FLOAT32);
    
    // Initialize skip connection D
    std::vector<int64_t> D_shape = {d_model};
    D_ = std::make_unique<Tensor>(D_shape, core::DataType::FLOAT32);
    
    // Initialize discretization parameter delta
    std::vector<int64_t> delta_shape = {d_model};
    delta_ = std::make_unique<Tensor>(delta_shape, core::DataType::FLOAT32);
    
    // Random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);
    
    auto init_tensor = [&](Tensor& tensor) {
        float* data = tensor.data_ptr<float>();
        for (int64_t i = 0; i < tensor.numel(); ++i) {
            data[i] = dist(gen);
        }
    };
    
    init_tensor(*A_);
    init_tensor(*B_);
    init_tensor(*C_);
    init_tensor(*D_);
    
    // Initialize delta with small positive values
    float* delta_data = delta_->data_ptr<float>();
    std::uniform_real_distribution<float> delta_dist(config_.dt_min, config_.dt_max);
    for (int64_t i = 0; i < d_model; ++i) {
        delta_data[i] = delta_dist(gen);
    }
    
    // Initialize hidden state
    std::vector<int64_t> state_shape = {1, d_state}; // Will be resized as needed
    hidden_state_ = std::make_unique<Tensor>(state_shape, core::DataType::FLOAT32);
    reset_state();
}

Tensor MambaSSM::forward(const Tensor& input, const Tensor* state) {
    const auto& input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t seq_len = input_shape[1];
    int64_t d_model = input_shape[2];
    
    Tensor output(input_shape, input.dtype());
    
    // Resize hidden state if needed
    if (hidden_state_->shape()[0] != batch_size) {
        std::vector<int64_t> state_shape = {batch_size, config_.d_state};
        hidden_state_ = std::make_unique<Tensor>(state_shape, core::DataType::FLOAT32);
        reset_state();
    }
    
    // Use provided state if available
    if (state) {
        *hidden_state_ = *state;
    }
    
    selective_scan_kernel(
        input.data_ptr<float>(), delta_->data_ptr<float>(),
        A_->data_ptr<float>(), B_->data_ptr<float>(), C_->data_ptr<float>(),
        output.data_ptr<float>(), hidden_state_->data_ptr<float>(),
        batch_size, seq_len, d_model, config_.d_state
    );
    
    return output;
}

void MambaSSM::selective_scan_kernel(
    const float* input_ptr, const float* delta_ptr,
    const float* A_ptr, const float* B_ptr, const float* C_ptr,
    float* output_ptr, float* state_ptr,
    int64_t batch_size, int64_t seq_len, int64_t d_model, int64_t d_state) {
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t t = 0; t < seq_len; ++t) {
            
            // For each model dimension
            for (int64_t d = 0; d < d_model; ++d) {
                float x_t = input_ptr[b * seq_len * d_model + t * d_model + d];
                float dt = delta_ptr[d];
                
                // Update state for this dimension
                for (int64_t s = 0; s < d_state; ++s) {
                    int64_t state_idx = b * d_state + s;
                    int64_t A_idx = s * d_model + d;
                    int64_t B_idx = s * d_model + d;
                    
                    // Discretize: h_t = exp(dt * A) * h_{t-1} + dt * B * x_t
                    float A_discrete = std::exp(dt * A_ptr[A_idx]);
                    state_ptr[state_idx] = A_discrete * state_ptr[state_idx] + dt * B_ptr[B_idx] * x_t;
                }
                
                // Compute output: y_t = C * h_t + D * x_t
                float y_t = D_->data_ptr<float>()[d] * x_t;
                for (int64_t s = 0; s < d_state; ++s) {
                    int64_t state_idx = b * d_state + s;
                    int64_t C_idx = d * d_state + s;
                    y_t += C_ptr[C_idx] * state_ptr[state_idx];
                }
                
                output_ptr[b * seq_len * d_model + t * d_model + d] = y_t;
            }
        }
    }
}

std::vector<Tensor> MambaSSM::get_state() const {
    return {*hidden_state_};
}

void MambaSSM::reset_state() {
    float* state_data = hidden_state_->data_ptr<float>();
    std::fill(state_data, state_data + hidden_state_->numel(), 0.0f);
}

size_t MambaSSM::estimate_memory_usage(int64_t seq_len, int64_t batch_size) const {
    return config_.d_model * config_.d_state * sizeof(float) * 3 + // A, B, C
           config_.d_model * sizeof(float) * 2 + // D, delta
           batch_size * config_.d_state * sizeof(float) + // hidden state
           batch_size * seq_len * config_.d_model * sizeof(float); // output
}

void MambaSSM::set_selective_parameters(const Tensor& delta, const Tensor& A, const Tensor& B, const Tensor& C) {
    *delta_ = delta;
    *A_ = A;
    *B_ = B;
    *C_ = C;
}

Tensor MambaSSM::selective_scan(const Tensor& input, const Tensor& delta, 
                               const Tensor& A, const Tensor& B, const Tensor& C) {
    // Temporarily set parameters
    auto old_delta = *delta_;
    auto old_A = *A_;
    auto old_B = *B_;
    auto old_C = *C_;
    
    set_selective_parameters(delta, A, B, C);
    auto result = forward(input);
    
    // Restore old parameters
    *delta_ = old_delta;
    *A_ = old_A;
    *B_ = old_B;
    *C_ = old_C;
    
    return result;
}

void MambaSSM::discretize_bilinear(const Tensor& delta, const Tensor& A, const Tensor& B,
                                  Tensor& A_discrete, Tensor& B_discrete) {
    // Bilinear transform: (I - dt/2 * A)^{-1} * (I + dt/2 * A)
    // Simplified implementation for diagonal A
    const float* delta_data = delta.data_ptr<float>();
    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* A_disc_data = A_discrete.data_ptr<float>();
    float* B_disc_data = B_discrete.data_ptr<float>();
    
    for (int64_t i = 0; i < A.numel(); ++i) {
        float dt = delta_data[i % delta.numel()];
        float a = A_data[i];
        A_disc_data[i] = (1.0f + dt * a * 0.5f) / (1.0f - dt * a * 0.5f);
        B_disc_data[i] = dt * B_data[i] / (1.0f - dt * a * 0.5f);
    }
}

void MambaSSM::discretize_zoh(const Tensor& delta, const Tensor& A, const Tensor& B,
                             Tensor& A_discrete, Tensor& B_discrete) {
    // Zero-order hold: A_d = exp(dt * A), B_d = A^{-1} * (A_d - I) * B
    const float* delta_data = delta.data_ptr<float>();
    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* A_disc_data = A_discrete.data_ptr<float>();
    float* B_disc_data = B_discrete.data_ptr<float>();
    
    for (int64_t i = 0; i < A.numel(); ++i) {
        float dt = delta_data[i % delta.numel()];
        float a = A_data[i];
        A_disc_data[i] = std::exp(dt * a);
        if (std::abs(a) > 1e-6f) {
            B_disc_data[i] = (A_disc_data[i] - 1.0f) / a * B_data[i];
        } else {
            B_disc_data[i] = dt * B_data[i]; // Limit as a -> 0
        }
    }
}

// S4SSM Implementation
S4SSM::S4SSM(const S4Config& config) : StateSpaceModelBase(config), s4_config_(config) {
    initialize_structured_matrix();
}

void S4SSM::initialize_structured_matrix() {
    int64_t d_model = config_.d_model;
    int64_t d_state = config_.d_state;
    int64_t rank = s4_config_.rank;
    
    // Initialize diagonal eigenvalues
    std::vector<int64_t> lambda_shape = {d_state};
    Lambda_ = std::make_unique<Tensor>(lambda_shape, core::DataType::FLOAT32);
    
    // Initialize low-rank factors
    std::vector<int64_t> P_shape = {d_state, rank};
    std::vector<int64_t> Q_shape = {rank, d_state};
    P_ = std::make_unique<Tensor>(P_shape, core::DataType::FLOAT32);
    Q_ = std::make_unique<Tensor>(Q_shape, core::DataType::FLOAT32);
    
    // Initialize B and C matrices
    std::vector<int64_t> B_shape = {d_state, d_model};
    std::vector<int64_t> C_shape = {d_model, d_state};
    B_ = std::make_unique<Tensor>(B_shape, core::DataType::FLOAT32);
    C_ = std::make_unique<Tensor>(C_shape, core::DataType::FLOAT32);
    
    // Initialize discretization step
    std::vector<int64_t> dt_shape = {d_model};
    dt_ = std::make_unique<Tensor>(dt_shape, core::DataType::FLOAT32);
    
    // Initialize with HiPPO if specified
    if (s4_config_.structure == S4SSM::StructureType::HIPPO) {
        initialize_hippo_legs();
    } else {
        // Random initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        auto init_tensor = [&](Tensor& tensor) {
            float* data = tensor.data_ptr<float>();
            for (int64_t i = 0; i < tensor.numel(); ++i) {
                data[i] = dist(gen);
            }
        };
        
        init_tensor(*Lambda_);
        init_tensor(*P_);
        init_tensor(*Q_);
        init_tensor(*B_);
        init_tensor(*C_);
        
        // Initialize dt with small positive values
        float* dt_data = dt_->data_ptr<float>();
        std::uniform_real_distribution<float> dt_dist(config_.dt_min, config_.dt_max);
        for (int64_t i = 0; i < d_model; ++i) {
            dt_data[i] = dt_dist(gen);
        }
    }
}

void S4SSM::initialize_hippo_legs() {
    // Initialize with HiPPO-LegS matrix
    int64_t N = config_.d_state;
    float* lambda_data = Lambda_->data_ptr<float>();
    
    for (int64_t i = 0; i < N; ++i) {
        lambda_data[i] = -(i + 1); // Negative eigenvalues for stability
    }
    
    // Initialize B and C for HiPPO
    float* B_data = B_->data_ptr<float>();
    float* C_data = C_->data_ptr<float>();
    
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < config_.d_model; ++j) {
            B_data[i * config_.d_model + j] = std::sqrt(2 * i + 1);
            C_data[j * N + i] = std::sqrt(2 * i + 1);
        }
    }
}

Tensor S4SSM::forward(const Tensor& input, const Tensor* state) {
    const auto& input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t seq_len = input_shape[1];
    int64_t d_model = input_shape[2];
    
    if (s4_config_.use_fft && seq_len > 64) {
        // Use convolution mode for long sequences
        auto kernel = compute_convolution_kernel(seq_len);
        return fft_convolution(input, kernel);
    } else {
        // Use recurrent mode for short sequences
        Tensor output(input_shape, input.dtype());
        
        // Initialize state if not provided
        std::vector<int64_t> state_shape = {batch_size, config_.d_state};
        Tensor internal_state(state_shape, core::DataType::FLOAT32);
        if (state) {
            internal_state = *state;
        } else {
            float* state_data = internal_state.data_ptr<float>();
            std::fill(state_data, state_data + internal_state.numel(), 0.0f);
        }
        
        s4_recurrent_kernel(
            input.data_ptr<float>(), output.data_ptr<float>(), internal_state.data_ptr<float>(),
            batch_size, seq_len, d_model, config_.d_state
        );
        
        return output;
    }
}

Tensor S4SSM::compute_convolution_kernel(int64_t seq_len) {
    // Check cache first
    auto it = kernel_cache_.find(seq_len);
    if (it != kernel_cache_.end()) {
        return it->second;
    }
    
    // Compute kernel: K = C * (I - A)^{-1} * B
    std::vector<int64_t> kernel_shape = {seq_len, config_.d_model};
    Tensor kernel(kernel_shape, core::DataType::FLOAT32);
    
    // Simplified kernel computation for diagonal + low-rank structure
    float* kernel_data = kernel.data_ptr<float>();
    const float* lambda_data = Lambda_->data_ptr<float>();
    const float* B_data = B_->data_ptr<float>();
    const float* C_data = C_->data_ptr<float>();
    const float* dt_data = dt_->data_ptr<float>();
    
    for (int64_t t = 0; t < seq_len; ++t) {
        for (int64_t d = 0; d < config_.d_model; ++d) {
            float sum = 0.0f;
            float dt = dt_data[d];
            
            for (int64_t s = 0; s < config_.d_state; ++s) {
                float lambda_discrete = std::exp(dt * lambda_data[s]);
                float power = std::pow(lambda_discrete, t);
                sum += C_data[d * config_.d_state + s] * power * B_data[s * config_.d_model + d];
            }
            
            kernel_data[t * config_.d_model + d] = sum;
        }
    }
    
    // Cache the kernel
    kernel_cache_[seq_len] = kernel;
    return kernel;
}

Tensor S4SSM::fft_convolution(const Tensor& input, const Tensor& kernel) {
    // Simplified FFT convolution implementation
    const auto& input_shape = input.shape();
    Tensor output(input_shape, input.dtype());
    
    s4_convolution_kernel(
        input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(),
        input_shape[0], input_shape[1], input_shape[2]
    );
    
    return output;
}

void S4SSM::s4_recurrent_kernel(
    const float* input_ptr, float* output_ptr, float* state_ptr,
    int64_t batch_size, int64_t seq_len, int64_t d_model, int64_t d_state) {
    
    const float* lambda_data = Lambda_->data_ptr<float>();
    const float* B_data = B_->data_ptr<float>();
    const float* C_data = C_->data_ptr<float>();
    const float* dt_data = dt_->data_ptr<float>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t t = 0; t < seq_len; ++t) {
            for (int64_t d = 0; d < d_model; ++d) {
                float x_t = input_ptr[b * seq_len * d_model + t * d_model + d];
                float dt = dt_data[d];
                
                // Update state: h_t = A_discrete * h_{t-1} + B_discrete * x_t
                for (int64_t s = 0; s < d_state; ++s) {
                    int64_t state_idx = b * d_state + s;
                    float lambda_discrete = std::exp(dt * lambda_data[s]);
                    state_ptr[state_idx] = lambda_discrete * state_ptr[state_idx] + 
                                          dt * B_data[s * d_model + d] * x_t;
                }
                
                // Compute output: y_t = C * h_t
                float y_t = 0.0f;
                for (int64_t s = 0; s < d_state; ++s) {
                    int64_t state_idx = b * d_state + s;
                    y_t += C_data[d * d_state + s] * state_ptr[state_idx];
                }
                
                output_ptr[b * seq_len * d_model + t * d_model + d] = y_t;
            }
        }
    }
}

void S4SSM::s4_convolution_kernel(
    const float* input_ptr, const float* kernel_ptr, float* output_ptr,
    int64_t batch_size, int64_t seq_len, int64_t d_model) {
    
    // Simple causal convolution
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t t = 0; t < seq_len; ++t) {
            for (int64_t d = 0; d < d_model; ++d) {
                float sum = 0.0f;
                
                for (int64_t k = 0; k <= t; ++k) {
                    sum += input_ptr[b * seq_len * d_model + k * d_model + d] * 
                           kernel_ptr[(t - k) * d_model + d];
                }
                
                output_ptr[b * seq_len * d_model + t * d_model + d] = sum;
            }
        }
    }
}

std::vector<Tensor> S4SSM::get_state() const {
    // Return empty state for stateless S4
    return {};
}

void S4SSM::reset_state() {
    // S4 is typically stateless in convolution mode
    kernel_cache_.clear();
}

size_t S4SSM::estimate_memory_usage(int64_t seq_len, int64_t batch_size) const {
    return config_.d_state * sizeof(float) + // Lambda
           config_.d_state * s4_config_.rank * sizeof(float) * 2 + // P, Q
           config_.d_state * config_.d_model * sizeof(float) * 2 + // B, C
           config_.d_model * sizeof(float) + // dt
           seq_len * config_.d_model * sizeof(float); // kernel cache
}

// S5SSM Implementation
S5SSM::S5SSM(const Config& config) : StateSpaceModelBase(config) {
    initialize_diagonal_parameters();
}

void S5SSM::initialize_diagonal_parameters() {
    int64_t d_model = config_.d_model;
    int64_t d_state = config_.d_state;
    
    // Initialize diagonal state matrix
    std::vector<int64_t> lambda_shape = {d_state};
    Lambda_ = std::make_unique<Tensor>(lambda_shape, core::DataType::FLOAT32);
    
    // Initialize B and C matrices
    std::vector<int64_t> B_shape = {d_state, d_model};
    std::vector<int64_t> C_shape = {d_model, d_state};
    B_ = std::make_unique<Tensor>(B_shape, core::DataType::FLOAT32);
    C_ = std::make_unique<Tensor>(C_shape, core::DataType::FLOAT32);
    
    // Initialize discretization step
    std::vector<int64_t> dt_shape = {d_model};
    dt_ = std::make_unique<Tensor>(dt_shape, core::DataType::FLOAT32);
    
    // Random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    auto init_tensor = [&](Tensor& tensor) {
        float* data = tensor.data_ptr<float>();
        for (int64_t i = 0; i < tensor.numel(); ++i) {
            data[i] = dist(gen);
        }
    };
    
    init_tensor(*Lambda_);
    init_tensor(*B_);
    init_tensor(*C_);
    
    // Initialize dt with small positive values
    float* dt_data = dt_->data_ptr<float>();
    std::uniform_real_distribution<float> dt_dist(config_.dt_min, config_.dt_max);
    for (int64_t i = 0; i < d_model; ++i) {
        dt_data[i] = dt_dist(gen);
    }
    
    // Initialize hidden state
    std::vector<int64_t> state_shape = {1, d_state};
    hidden_state_ = std::make_unique<Tensor>(state_shape, core::DataType::FLOAT32);
    reset_state();
}

Tensor S5SSM::forward(const Tensor& input, const Tensor* state) {
    return parallel_scan(input);
}

Tensor S5SSM::parallel_scan(const Tensor& input) {
    const auto& input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t seq_len = input_shape[1];
    int64_t d_model = input_shape[2];
    
    Tensor output(input_shape, input.dtype());
    
    // Allocate workspace for parallel scan
    std::vector<float> workspace(batch_size * seq_len * config_.d_state);
    
    s5_parallel_scan_kernel(
        input.data_ptr<float>(), output.data_ptr<float>(),
        Lambda_->data_ptr<float>(), B_->data_ptr<float>(), C_->data_ptr<float>(),
        workspace.data(), batch_size, seq_len, d_model, config_.d_state
    );
    
    return output;
}

void S5SSM::s5_parallel_scan_kernel(
    const float* input_ptr, float* output_ptr,
    const float* lambda_ptr, const float* B_ptr, const float* C_ptr,
    float* workspace_ptr, int64_t batch_size, int64_t seq_len, 
    int64_t d_model, int64_t d_state) {
    
    const float* dt_data = dt_->data_ptr<float>();
    
    // Simplified parallel scan (sequential for now)
    for (int64_t b = 0; b < batch_size; ++b) {
        // Initialize state
        std::vector<float> state(d_state, 0.0f);
        
        for (int64_t t = 0; t < seq_len; ++t) {
            for (int64_t d = 0; d < d_model; ++d) {
                float x_t = input_ptr[b * seq_len * d_model + t * d_model + d];
                float dt = dt_data[d];
                
                // Update state for each state dimension
                for (int64_t s = 0; s < d_state; ++s) {
                    float lambda_discrete = std::exp(dt * lambda_ptr[s]);
                    state[s] = lambda_discrete * state[s] + dt * B_ptr[s * d_model + d] * x_t;
                }
                
                // Compute output
                float y_t = 0.0f;
                for (int64_t s = 0; s < d_state; ++s) {
                    y_t += C_ptr[d * d_state + s] * state[s];
                }
                
                output_ptr[b * seq_len * d_model + t * d_model + d] = y_t;
            }
        }
    }
}

std::vector<Tensor> S5SSM::get_state() const {
    return {*hidden_state_};
}

void S5SSM::reset_state() {
    float* state_data = hidden_state_->data_ptr<float>();
    std::fill(state_data, state_data + hidden_state_->numel(), 0.0f);
}

size_t S5SSM::estimate_memory_usage(int64_t seq_len, int64_t batch_size) const {
    return config_.d_state * sizeof(float) + // Lambda
           config_.d_state * config_.d_model * sizeof(float) * 2 + // B, C
           config_.d_model * sizeof(float) + // dt
           batch_size * config_.d_state * sizeof(float) + // hidden state
           batch_size * seq_len * config_.d_state * sizeof(float); // workspace
}

// Factory functions
void SSMFactory::benchmark_ssm_variants(int64_t seq_len, int64_t d_model, int64_t batch_size) {
    std::cout << "SSM benchmarking not implemented yet\n";
}

void SSMFactory::compare_memory_usage(int64_t seq_len, int64_t d_model) {
    std::cout << "SSM memory comparison not implemented yet\n";
}

} // namespace models
} // namespace operators
} // namespace deepcpp 