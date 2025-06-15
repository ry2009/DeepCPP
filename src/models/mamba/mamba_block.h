#pragma once

#include "../../core/tensor/tensor.h"
#include "../../operators/activations/activations.h"
#include <memory>
#include <vector>
#include <complex>

namespace deepcpp {
namespace models {
namespace mamba {

using namespace deepcpp::core;
using namespace deepcpp::operators;

/**
 * Selective State Space Model (S6) - Core of Mamba Architecture
 * 
 * Implements the selective SSM that can selectively remember or forget
 * information based on input, making it more capable than traditional SSMs.
 * 
 * Paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
 */
class SelectiveSSM {
public:
    struct Config {
        int64_t d_model = 768;           // Model dimension
        int64_t d_state = 16;            // SSM state dimension (N)
        int64_t d_conv = 4;              // Local convolution width
        int64_t expand = 2;              // Block expansion factor
        float dt_rank_ratio = -1.0f;    // dt rank ratio (auto if -1)
        int64_t dt_min_exp = -3;         // Minimum dt exponent
        int64_t dt_max_exp = 3;          // Maximum dt exponent
        float dt_init_floor = 1e-4f;     // Floor value for dt initialization
        bool bias = false;               // Use bias in linear layers
        std::string activation = "silu"; // Activation function
        bool use_fast_path = true;       // Use optimized kernels when possible
    };
    
    SelectiveSSM(const Config& config);
    ~SelectiveSSM();
    
    /**
     * Forward pass of Selective SSM
     * 
     * @param x Input tensor [batch, seq_len, d_model]
     * @return Output tensor [batch, seq_len, d_model]
     */
    Tensor forward(const Tensor& x);
    
    /**
     * Forward pass with state caching for inference
     * 
     * @param x Input tensor [batch, seq_len, d_model]
     * @param state Previous state [batch, d_inner, d_state]
     * @return {output, new_state}
     */
    std::pair<Tensor, Tensor> forward_with_cache(const Tensor& x, const Tensor& state);
    
    /**
     * Recurrent forward pass (step-by-step)
     * 
     * @param x Input tensor [batch, 1, d_model] (single timestep)
     * @param state Previous state [batch, d_inner, d_state]
     * @return {output, new_state}
     */
    std::pair<Tensor, Tensor> step(const Tensor& x, const Tensor& state);
    
    // Configuration
    const Config& get_config() const { return config_; }
    
private:
    Config config_;
    int64_t d_inner_;  // expand * d_model
    int64_t dt_rank_;  // rank of dt projection
    
    // Learnable parameters
    Tensor in_proj_;      // [d_model, d_inner * 2]
    Tensor conv1d_weight_; // [d_inner, 1, d_conv]
    Tensor conv1d_bias_;   // [d_inner]
    Tensor x_proj_;       // [d_inner, dt_rank + d_state * 2]
    Tensor dt_proj_;      // [dt_rank, d_inner]
    Tensor A_log_;        // [d_inner, d_state] (in log space)
    Tensor D_;            // [d_inner]
    Tensor out_proj_;     // [d_inner, d_model]
    
    // Precomputed values
    Tensor A_;            // [d_inner, d_state] (actual A matrix)
    
    // Core SSM operations
    Tensor selective_scan(const Tensor& u, const Tensor& delta, 
                         const Tensor& A, const Tensor& B, const Tensor& C,
                         const Tensor& D, const Tensor& z = Tensor());
    
    Tensor selective_scan_parallel(const Tensor& u, const Tensor& delta,
                                  const Tensor& A, const Tensor& B, const Tensor& C,
                                  const Tensor& D, const Tensor& z = Tensor());
    
    Tensor selective_scan_recurrent(const Tensor& u, const Tensor& delta,
                                   const Tensor& A, const Tensor& B, const Tensor& C,
                                   const Tensor& D, const Tensor& z = Tensor());
    
    // Initialization helpers
    void init_parameters();
    void init_A_matrix();
    void init_dt_projection();
    
    // Optimized kernels
    void selective_scan_kernel_cuda(
        const float* u_ptr, const float* delta_ptr, const float* A_ptr,
        const float* B_ptr, const float* C_ptr, const float* D_ptr,
        float* y_ptr, float* state_ptr,
        int64_t batch, int64_t seq_len, int64_t d_inner, int64_t d_state
    );
    
    void selective_scan_kernel_cpu(
        const float* u_ptr, const float* delta_ptr, const float* A_ptr,
        const float* B_ptr, const float* C_ptr, const float* D_ptr,
        float* y_ptr, float* state_ptr,
        int64_t batch, int64_t seq_len, int64_t d_inner, int64_t d_state
    );
    
    // Parallel scan implementation
    void parallel_scan_kernel(
        const float* u_ptr, const float* delta_ptr, const float* A_ptr,
        const float* B_ptr, const float* C_ptr,
        float* y_ptr, float* states_ptr,
        int64_t batch, int64_t seq_len, int64_t d_inner, int64_t d_state
    );
    
    // Memory management
    std::unique_ptr<float[]> workspace_;
    size_t workspace_size_;
    void ensure_workspace(size_t required_size);
};

/**
 * Mamba Block - Complete Mamba layer with residual connections
 */
class MambaBlock {
public:
    struct Config {
        int64_t d_model = 768;
        int64_t d_state = 16;
        int64_t d_conv = 4;
        int64_t expand = 2;
        float dt_rank_ratio = -1.0f;
        bool use_norm = true;
        std::string norm_type = "layer";  // "layer" or "rms"
        float norm_eps = 1e-5f;
        bool use_bias = false;
        std::string activation = "silu";
        float dropout = 0.0f;
    };
    
    MambaBlock(const Config& config);
    
    /**
     * Forward pass through Mamba block
     * 
     * @param x Input tensor [batch, seq_len, d_model]
     * @return Output tensor [batch, seq_len, d_model]
     */
    Tensor forward(const Tensor& x);
    
    /**
     * Forward with state caching for inference
     */
    std::pair<Tensor, Tensor> forward_with_cache(const Tensor& x, const Tensor& state);
    
    const Config& get_config() const { return config_; }
    
private:
    Config config_;
    std::unique_ptr<SelectiveSSM> ssm_;
    
    // Normalization layers
    Tensor norm_weight_;
    Tensor norm_bias_;
    
    // Dropout for training
    float dropout_prob_;
    
    // Normalization functions
    Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps);
    Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps);
};

/**
 * Bidirectional Mamba - Processes sequences in both directions
 */
class BidirectionalMamba {
public:
    struct Config {
        MambaBlock::Config forward_config;
        MambaBlock::Config backward_config;
        std::string combination = "concat";  // "concat", "add", "gated"
        int64_t d_model = 768;
    };
    
    BidirectionalMamba(const Config& config);
    
    Tensor forward(const Tensor& x);
    
private:
    Config config_;
    std::unique_ptr<MambaBlock> forward_mamba_;
    std::unique_ptr<MambaBlock> backward_mamba_;
    
    // For gated combination
    Tensor gate_proj_;
    
    Tensor reverse_sequence(const Tensor& x);
    Tensor combine_bidirectional(const Tensor& forward_out, const Tensor& backward_out);
};

/**
 * Multi-dimensional Mamba for 2D/3D data
 */
class MultiDimensionalMamba {
public:
    struct Config {
        std::vector<int64_t> dims = {768};    // Dimensions for each axis
        std::vector<int64_t> d_states = {16}; // State dims for each axis
        int64_t d_conv = 4;
        int64_t expand = 2;
        std::string scan_order = "raster";     // "raster", "hilbert", "snake"
    };
    
    MultiDimensionalMamba(const Config& config);
    
    /**
     * Forward pass for 2D data (images)
     * 
     * @param x Input tensor [batch, height, width, channels]
     * @return Output tensor [batch, height, width, channels]
     */
    Tensor forward_2d(const Tensor& x);
    
    /**
     * Forward pass for 3D data (videos)
     * 
     * @param x Input tensor [batch, time, height, width, channels]
     * @return Output tensor [batch, time, height, width, channels]
     */
    Tensor forward_3d(const Tensor& x);
    
private:
    Config config_;
    std::vector<std::unique_ptr<SelectiveSSM>> ssm_layers_;
    
    // Scanning patterns
    std::vector<int64_t> get_raster_order(const std::vector<int64_t>& shape);
    std::vector<int64_t> get_hilbert_order(int64_t height, int64_t width);
    std::vector<int64_t> get_snake_order(int64_t height, int64_t width);
    
    Tensor apply_scanning_pattern(const Tensor& x, const std::vector<int64_t>& order);
    Tensor reverse_scanning_pattern(const Tensor& x, const std::vector<int64_t>& order,
                                   const std::vector<int64_t>& original_shape);
};

/**
 * Mamba with Linear Attention Hybrid
 */
class MambaLinearAttentionHybrid {
public:
    struct Config {
        int64_t d_model = 768;
        int64_t num_heads = 12;
        int64_t head_dim = 64;
        MambaBlock::Config mamba_config;
        float mamba_weight = 0.5f;       // Weight for combining mamba and attention
        std::string combination = "weighted"; // "weighted", "gated", "parallel"
    };
    
    MambaLinearAttentionHybrid(const Config& config);
    
    Tensor forward(const Tensor& x);
    
private:
    Config config_;
    std::unique_ptr<MambaBlock> mamba_block_;
    
    // Linear attention components
    Tensor q_proj_, k_proj_, v_proj_, o_proj_;
    
    // Combination weights/gates
    Tensor combination_weight_;
    Tensor combination_gate_;
    
    Tensor linear_attention(const Tensor& x);
    Tensor combine_outputs(const Tensor& mamba_out, const Tensor& attention_out);
};

/**
 * Mamba Model - Complete model with multiple Mamba blocks
 */
class MambaModel {
public:
    struct Config {
        int64_t vocab_size = 50257;
        int64_t d_model = 768;
        int64_t n_layers = 12;
        MambaBlock::Config block_config;
        bool tie_embeddings = true;
        float pad_vocab_to_multiple = 1.0f;
        bool use_pos_emb = false;        // Mamba typically doesn't need positional embeddings
        int64_t max_seq_len = 2048;
    };
    
    MambaModel(const Config& config);
    
    /**
     * Forward pass through complete Mamba model
     * 
     * @param input_ids Token IDs [batch, seq_len]
     * @return Logits [batch, seq_len, vocab_size]
     */
    Tensor forward(const Tensor& input_ids);
    
    /**
     * Generation with KV caching
     * 
     * @param input_ids Initial tokens [batch, seq_len]
     * @param max_new_tokens Maximum tokens to generate
     * @param temperature Sampling temperature
     * @param top_k Top-k sampling
     * @param top_p Top-p (nucleus) sampling
     * @return Generated tokens [batch, seq_len + max_new_tokens]
     */
    Tensor generate(const Tensor& input_ids, int64_t max_new_tokens,
                   float temperature = 1.0f, int64_t top_k = 50, float top_p = 0.95f);
    
    /**
     * Streaming generation (one token at a time)
     */
    class StreamingGenerator {
    public:
        StreamingGenerator(MambaModel* model, const Tensor& initial_tokens);
        ~StreamingGenerator();
        
        Tensor next_token(float temperature = 1.0f, int64_t top_k = 50, float top_p = 0.95f);
        bool is_finished() const { return finished_; }
        const Tensor& get_generated_tokens() const { return generated_tokens_; }
        
    private:
        MambaModel* model_;
        std::vector<Tensor> layer_states_;
        Tensor generated_tokens_;
        bool finished_;
    };
    
    std::unique_ptr<StreamingGenerator> create_generator(const Tensor& initial_tokens);
    
    const Config& get_config() const { return config_; }
    
private:
    Config config_;
    
    // Model components
    Tensor embedding_;                    // [vocab_size, d_model]
    std::vector<std::unique_ptr<MambaBlock>> layers_;
    Tensor final_norm_weight_;
    Tensor final_norm_bias_;
    Tensor lm_head_;                     // [d_model, vocab_size] (tied with embedding if specified)
    
    // For generation
    std::vector<Tensor> init_layer_states(int64_t batch_size);
    Tensor sample_token(const Tensor& logits, float temperature, int64_t top_k, float top_p);
    
    // Embedding helpers
    void init_embeddings();
    void tie_weights();
};

/**
 * Optimized Mamba kernels and utilities
 */
namespace kernels {

/**
 * Fused selective scan kernel with various optimizations
 */
void fused_selective_scan(
    const float* u, const float* delta, const float* A, const float* B, const float* C,
    const float* D, const float* z, float* y, float* last_state,
    int64_t batch, int64_t seq_len, int64_t d_inner, int64_t d_state,
    bool has_D = true, bool has_z = false, bool return_last_state = false
);

/**
 * Optimized convolution for Mamba's 1D conv
 */
void conv1d_fused_activation(
    const float* x, const float* weight, const float* bias,
    float* y, int64_t batch, int64_t seq_len, int64_t d_inner, int64_t kernel_size,
    const std::string& activation = "silu"
);

/**
 * Parallel prefix sum for efficient SSM computation
 */
void parallel_prefix_sum(
    const float* input, float* output, int64_t length,
    std::function<void(float&, const float&)> combine_fn
);

/**
 * Optimized matrix multiplication for SSM projections
 */
void ssm_linear_projection(
    const float* x, const float* weight, const float* bias,
    float* y, int64_t batch, int64_t seq_len, int64_t in_dim, int64_t out_dim
);

} // namespace kernels

} // namespace mamba
} // namespace models
} // namespace deepcpp 