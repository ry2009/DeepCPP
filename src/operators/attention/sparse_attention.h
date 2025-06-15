#pragma once

#include "../../core/tensor/tensor.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <immintrin.h>

namespace deepcpp {
namespace operators {
namespace attention {

// Forward declarations
using deepcpp::core::Tensor;
using deepcpp::core::DataType;

/**
 * Sparse Attention Patterns and Kernels
 * 
 * Implements various sparse attention mechanisms for efficient processing
 * of long sequences with reduced computational complexity.
 */

enum class SparsePattern {
    LOCAL,           // Local sliding window attention
    STRIDED,         // Strided attention with fixed stride
    DILATED,         // Dilated attention with increasing dilation
    BIGBIRD,         // BigBird: Random + Global + Local
    LONGFORMER,      // Longformer: Local + Global tokens
    REFORMER,        // Reformer: LSH attention
    LINFORMER,       // Linformer: Low-rank projections
    PERFORMER,       // Performer: FAVOR+ kernel approximation
    CUSTOM           // Custom user-defined pattern
};

/**
 * Sparse Attention Block Mask
 * 
 * Represents which attention blocks are computed vs. masked out
 */
struct SparseBlockMask {
    int64_t seq_len;
    int64_t block_size;
    int64_t num_blocks;
    std::vector<std::vector<bool>> block_mask;  // [num_blocks][num_blocks]
    
    // Default constructor for containers
    SparseBlockMask() : seq_len(0), block_size(0), num_blocks(0) {}
    
    SparseBlockMask(int64_t seq_len, int64_t block_size);
    
    // Pattern generators
    void generate_local_pattern(int64_t window_size);
    void generate_strided_pattern(int64_t stride);
    void generate_dilated_pattern(const std::vector<int64_t>& dilations);
    void generate_bigbird_pattern(int64_t num_random_blocks, int64_t local_window);
    void generate_longformer_pattern(int64_t local_window, 
                                   const std::vector<int64_t>& global_positions);
    void generate_custom_pattern(const std::vector<std::vector<bool>>& custom_mask);
    
    // Utilities
    int64_t count_active_blocks() const;
    float sparsity_ratio() const;
    void print_pattern() const;
};

/**
 * Local Attention (Sliding Window)
 * 
 * Each token attends to a fixed-size local window
 */
class LocalAttention {
public:
    struct Config {
        int64_t window_size;
        bool bidirectional;
        float scale;
        
        Config(int64_t ws = 256, bool bidir = true, float s = -1.0f)
            : window_size(ws), bidirectional(bidir), scale(s) {}
    };
    
    LocalAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v);
    
    // Memory usage estimation
    size_t estimate_memory_usage(int64_t seq_len, int64_t head_dim) const;
    
private:
    Config config_;
    
    void local_attention_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len, 
        int64_t num_heads, int64_t head_dim
    );
};

/**
 * Strided Attention
 * 
 * Tokens attend to every stride-th position
 */
class StridedAttention {
public:
    struct Config {
        int64_t stride;
        int64_t local_window;  // Additional local context
        float scale;
        
        Config(int64_t s = 64, int64_t lw = 32, float sc = -1.0f)
            : stride(s), local_window(lw), scale(sc) {}
    };
    
    StridedAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v);
    
private:
    Config config_;
    
    void strided_attention_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len,
        int64_t num_heads, int64_t head_dim
    );
};

/**
 * BigBird Attention
 * 
 * Combines random attention, global tokens, and local sliding window
 */
class BigBirdAttention {
public:
    struct Config {
        int64_t block_size;
        int64_t num_random_blocks;
        int64_t window_size;
        int64_t num_global_blocks;
        float scale;
        int64_t random_seed;
        
        Config(int64_t bs = 64, int64_t nrb = 3, int64_t ws = 3, 
               int64_t ngb = 1, float s = -1.0f, int64_t seed = 42)
            : block_size(bs), num_random_blocks(nrb), window_size(ws),
              num_global_blocks(ngb), scale(s), random_seed(seed) {}
    };
    
    BigBirdAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v);
    
    // Generate and cache attention patterns
    void precompute_attention_pattern(int64_t seq_len);
    
private:
    Config config_;
    std::unordered_map<int64_t, SparseBlockMask> cached_patterns_;
    
    void bigbird_attention_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, const SparseBlockMask& mask,
        int64_t batch_size, int64_t seq_len, int64_t num_heads, int64_t head_dim
    );
};

/**
 * Longformer Attention
 * 
 * Sliding window + global attention for designated tokens
 */
class LongformerAttention {
public:
    struct Config {
        int64_t window_size;
        std::vector<int64_t> global_positions;  // Positions with global attention
        bool symmetric_window;
        float scale;
        
        Config(int64_t ws = 512, bool sym = true, float s = -1.0f)
            : window_size(ws), symmetric_window(sym), scale(s) {}
    };
    
    LongformerAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v);
    
    // Set global attention positions dynamically
    void set_global_positions(const std::vector<int64_t>& positions);
    
private:
    Config config_;
    
    void longformer_attention_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len,
        int64_t num_heads, int64_t head_dim
    );
    
    void compute_global_attention(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, const std::vector<int64_t>& global_pos,
        int64_t batch_size, int64_t seq_len, int64_t num_heads, int64_t head_dim
    );
};

/**
 * Block Sparse Attention
 * 
 * Generic implementation for any block-based sparsity pattern
 */
class BlockSparseAttention {
public:
    struct Config {
        int64_t block_size;
        SparsePattern pattern_type;
        float scale;
        
        // Pattern-specific parameters
        union {
            struct { int64_t window_size; } local;
            struct { int64_t stride; } strided;
            struct { int64_t num_random; int64_t seed; } bigbird;
        } pattern_params;
        
        Config(int64_t bs = 64, SparsePattern pt = SparsePattern::LOCAL, float s = -1.0f)
            : block_size(bs), pattern_type(pt), scale(s) {}
    };
    
    BlockSparseAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v,
                   const SparseBlockMask* custom_mask = nullptr);
    
    // Pattern management
    void set_sparse_pattern(const SparseBlockMask& mask);
    const SparseBlockMask& get_current_pattern() const;
    
private:
    Config config_;
    std::unique_ptr<SparseBlockMask> current_mask_;
    
    void block_sparse_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, const SparseBlockMask& mask,
        int64_t batch_size, int64_t seq_len, int64_t num_heads, int64_t head_dim
    );
    
    void compute_sparse_block(
        const float* q_block, const float* k_block, const float* v_block,
        float* output_block, int64_t block_q_size, int64_t block_kv_size,
        int64_t head_dim, float scale
    );
};

/**
 * Adaptive Sparse Attention
 * 
 * Dynamically adjusts sparsity pattern based on attention scores
 */
class AdaptiveSparseAttention {
public:
    struct Config {
        float sparsity_threshold;    // Top-k or threshold-based pruning
        int64_t min_attention_blocks;
        int64_t max_attention_blocks;
        bool use_topk;               // Use top-k vs threshold
        float scale;
        
        Config(float thresh = 0.1f, int64_t min_blocks = 1, int64_t max_blocks = 16,
               bool topk = true, float s = -1.0f)
            : sparsity_threshold(thresh), min_attention_blocks(min_blocks),
              max_attention_blocks(max_blocks), use_topk(topk), scale(s) {}
    };
    
    AdaptiveSparseAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v);
    
private:
    Config config_;
    
    // Two-pass algorithm: first pass to determine pattern, second pass to compute
    SparseBlockMask compute_adaptive_pattern(
        const float* q_ptr, const float* k_ptr,
        int64_t batch_size, int64_t seq_len, int64_t num_heads, int64_t head_dim
    );
    
    void adaptive_attention_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, const SparseBlockMask& adaptive_mask,
        int64_t batch_size, int64_t seq_len, int64_t num_heads, int64_t head_dim
    );
};

// Utility functions for sparse attention
namespace sparse_utils {

/**
 * Pattern Generators
 */
SparseBlockMask create_local_pattern(int64_t seq_len, int64_t block_size, int64_t window_size);
SparseBlockMask create_strided_pattern(int64_t seq_len, int64_t block_size, int64_t stride);
SparseBlockMask create_dilated_pattern(int64_t seq_len, int64_t block_size, 
                                     const std::vector<int64_t>& dilations);
SparseBlockMask create_bigbird_pattern(int64_t seq_len, int64_t block_size,
                                     int64_t num_random, int64_t window_size, int64_t seed);
SparseBlockMask create_longformer_pattern(int64_t seq_len, int64_t block_size,
                                        int64_t window_size, const std::vector<int64_t>& global_pos);

/**
 * Pattern Analysis
 */
float compute_sparsity_ratio(const SparseBlockMask& mask);
int64_t count_operations(const SparseBlockMask& mask, int64_t head_dim);
void visualize_pattern(const SparseBlockMask& mask, const std::string& filename = "");

/**
 * Pattern Optimization
 */
SparseBlockMask optimize_pattern_for_hardware(const SparseBlockMask& input_mask, 
                                            int64_t simd_width = 8);
SparseBlockMask merge_patterns(const std::vector<SparseBlockMask>& patterns);

} // namespace sparse_utils

} // namespace attention
} // namespace operators
} // namespace deepcpp 