#pragma once

#include "../../core/tensor/tensor.h"

using deepcpp::core::Tensor;
#include "../../core/memory/memory_pool.h"
#include <memory>
#include <cmath>
#include <immintrin.h>

namespace deepcpp {
namespace operators {
namespace attention {

using namespace deepcpp::core;

/**
 * Flash Attention Implementation
 * 
 * Memory-efficient attention computation that reduces memory complexity
 * from O(n²) to O(n) by computing attention in blocks and using online softmax.
 * 
 * Paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
 */
class FlashAttention {
public:
    struct Config {
        int64_t block_size_q;     // Query block size
        int64_t block_size_kv;    // Key/Value block size
        float scale;              // Attention scale (auto if negative)
        bool use_causal_mask;     // Enable causal masking
        bool enable_dropout;      // Enable dropout
        float dropout_prob;       // Dropout probability
        
        // Explicit constructor with default values
        Config(int64_t bsq = 64, int64_t bskv = 64, float s = -1.0f, 
               bool causal = false, bool dropout = false, float dropout_p = 0.1f) 
            : block_size_q(bsq), block_size_kv(bskv), scale(s), use_causal_mask(causal),
              enable_dropout(dropout), dropout_prob(dropout_p) {}
    };
    
    // Constructor with config
    explicit FlashAttention(const Config& config = Config());
    
    // Destructor
    ~FlashAttention();
    
    /**
     * Forward pass of Flash Attention
     * 
     * @param q Query tensor [batch, seq_len, num_heads, head_dim]
     * @param k Key tensor [batch, seq_len, num_heads, head_dim] 
     * @param v Value tensor [batch, seq_len, num_heads, head_dim]
     * @param mask Optional attention mask [batch, num_heads, seq_len, seq_len]
     * @return Attention output [batch, seq_len, num_heads, head_dim]
     */
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v, 
                   const Tensor& mask = Tensor());
    
    /**
     * Backward pass of Flash Attention
     * 
     * @param grad_output Gradient w.r.t output
     * @param q Query tensor from forward pass
     * @param k Key tensor from forward pass  
     * @param v Value tensor from forward pass
     * @param output Output from forward pass
     * @param lse Log-sum-exp from forward pass
     * @return Gradients {grad_q, grad_k, grad_v}
     */
    std::tuple<Tensor, Tensor, Tensor> backward(
        const Tensor& grad_output,
        const Tensor& q, const Tensor& k, const Tensor& v,
        const Tensor& output, const Tensor& lse
    );
    
    /**
     * Fused multi-head attention with flash attention
     */
    Tensor multi_head_attention(
        const Tensor& query,     // [batch, seq_len, d_model]
        const Tensor& key,       // [batch, seq_len, d_model]
        const Tensor& value,     // [batch, seq_len, d_model]
        const Tensor& w_q,       // [d_model, num_heads * head_dim]
        const Tensor& w_k,       // [d_model, num_heads * head_dim]
        const Tensor& w_v,       // [d_model, num_heads * head_dim]
        const Tensor& w_o,       // [num_heads * head_dim, d_model]
        const Tensor& mask = Tensor()
    );
    
    /**
     * Block-sparse flash attention for long sequences
     */
    Tensor block_sparse_attention(
        const Tensor& q, const Tensor& k, const Tensor& v,
        const std::vector<std::vector<int64_t>>& block_mask,
        int64_t block_size = 64
    );
    
    /**
     * Flash attention with relative position embeddings (RoPE)
     */
    Tensor flash_attention_with_rope(
        const Tensor& q, const Tensor& k, const Tensor& v,
        const Tensor& cos, const Tensor& sin,
        int64_t max_seq_len
    );
    
    /**
     * Flash attention with ALiBi (Attention with Linear Biases)
     */
    Tensor flash_attention_with_alibi(
        const Tensor& q, const Tensor& k, const Tensor& v,
        const std::vector<float>& alibi_slopes
    );
    
    // Configuration getters/setters
    const Config& get_config() const { return config_; }
    void set_config(const Config& config) { config_ = config; }
    
    // Memory usage estimation
    size_t estimate_memory_usage(int64_t batch_size, int64_t seq_len, 
                                 int64_t num_heads, int64_t head_dim) const;
    
private:
    Config config_;
    
    // Core flash attention kernels
    void flash_attention_forward_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, float* lse_ptr,
        int64_t batch_size, int64_t seq_len, int64_t num_heads, int64_t head_dim,
        const float* mask_ptr = nullptr
    );
    
    void flash_attention_backward_kernel(
        const float* grad_output_ptr, const float* q_ptr, const float* k_ptr, 
        const float* v_ptr, const float* output_ptr, const float* lse_ptr,
        float* grad_q_ptr, float* grad_k_ptr, float* grad_v_ptr,
        int64_t batch_size, int64_t seq_len, int64_t num_heads, int64_t head_dim
    );
    
    // Online softmax implementation
    struct OnlineSoftmaxState {
        __m256 max_val;
        __m256 sum_exp;
    };
    
    void online_softmax_update(OnlineSoftmaxState& state, const __m256& new_vals);
    __m256 online_softmax_finalize(const OnlineSoftmaxState& state);
    
    // Block processing functions
    void process_query_block(
        const float* q_block, const float* k_ptr, const float* v_ptr,
        float* output_block, float* lse_block,
        int64_t q_block_size, int64_t kv_seq_len, int64_t head_dim,
        int64_t kv_block_idx, const float* mask_ptr
    );
    
    void process_key_value_block(
        const float* q_ptr, const float* k_block, const float* v_block,
        float* output_ptr, float* lse_ptr, float* workspace,
        int64_t q_seq_len, int64_t kv_block_size, int64_t head_dim,
        int64_t q_block_idx, const float* mask_ptr
    );
    
    // Causal masking utilities
    void apply_causal_mask(float* scores, int64_t q_start, int64_t q_end,
                          int64_t k_start, int64_t k_end, int64_t seq_len);
    
    // Memory-efficient matrix operations
    void batched_gemm_strided(
        const float* a, const float* b, float* c,
        int64_t batch_size, int64_t m, int64_t n, int64_t k,
        int64_t stride_a, int64_t stride_b, int64_t stride_c,
        float alpha = 1.0f, float beta = 0.0f
    );
    
    // SIMD-optimized attention score computation
    void compute_attention_scores_simd(
        const float* q_ptr, const float* k_ptr, float* scores_ptr,
        int64_t q_len, int64_t k_len, int64_t head_dim, float scale
    );
    
    // Position encoding helpers
    void apply_rope_inplace(float* q_ptr, float* k_ptr, 
                           const float* cos_ptr, const float* sin_ptr,
                           int64_t seq_len, int64_t head_dim);
    
    void apply_alibi_bias(float* scores_ptr, int64_t q_len, int64_t k_len,
                         float alibi_slope, int64_t head_idx);
    
    // Workspace management
    std::unique_ptr<float[]> workspace_;
    size_t workspace_size_;
    void ensure_workspace_size(size_t required_size);
    
    // Thread-local storage for multi-threading
    thread_local static std::unique_ptr<float[]> thread_workspace_;
    thread_local static size_t thread_workspace_size_;
};

/**
 * Multi-Query Attention (MQA) with Flash Attention
 * 
 * Uses single key and value heads across all query heads for efficiency
 */
class FlashMultiQueryAttention {
public:
    struct Config {
        int64_t num_query_heads = 32;
        int64_t num_kv_heads = 1;        // MQA: single KV head
        int64_t head_dim = 64;
        int64_t block_size = 64;
        bool use_causal_mask = true;
        float scale = -1.0f;
    };
    
    FlashMultiQueryAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor& mask = Tensor());
    
private:
    Config config_;
    std::unique_ptr<FlashAttention> flash_attention_;
};

/**
 * Grouped-Query Attention (GQA) with Flash Attention
 * 
 * Groups multiple query heads to share fewer key-value heads
 */
class FlashGroupedQueryAttention {
public:
    struct Config {
        int64_t num_query_heads = 32;
        int64_t num_kv_heads = 8;        // GQA: reduced KV heads
        int64_t head_dim = 64;
        int64_t block_size = 64;
        bool use_causal_mask = true;
        float scale = -1.0f;
    };
    
    FlashGroupedQueryAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor& mask = Tensor());
    
private:
    Config config_;
    std::unique_ptr<FlashAttention> flash_attention_;
    int64_t group_size_;
};

/**
 * Flash Attention with Paged KV Cache
 * 
 * Optimized for inference with dynamic sequence lengths
 */
class PagedFlashAttention {
public:
    struct PagedKVCache {
        std::vector<Tensor> key_pages;
        std::vector<Tensor> value_pages;
        std::vector<int64_t> page_table;  // Maps logical to physical pages
        int64_t page_size;
        int64_t max_pages;
        int64_t current_length;
    };
    
    PagedFlashAttention(int64_t page_size = 64, int64_t max_pages = 1024);
    
    // Initialize KV cache
    void init_cache(PagedKVCache& cache, int64_t batch_size, 
                    int64_t num_heads, int64_t head_dim);
    
    // Append new key-value pairs to cache
    void append_kv(PagedKVCache& cache, const Tensor& new_k, const Tensor& new_v);
    
    // Attention with paged KV cache
    Tensor forward_with_cache(const Tensor& q, PagedKVCache& cache,
                             const Tensor& mask = Tensor());
    
private:
    int64_t page_size_;
    int64_t max_pages_;
    std::unique_ptr<FlashAttention> flash_attention_;
    
    void allocate_new_page(PagedKVCache& cache);
    void copy_kv_to_page(PagedKVCache& cache, const Tensor& k, const Tensor& v, 
                         int64_t page_idx, int64_t offset);
};

// Utility functions for attention patterns
namespace patterns {

/**
 * Generate attention patterns for different sparsity structures
 */
std::vector<std::vector<int64_t>> create_local_attention_pattern(
    int64_t seq_len, int64_t window_size);

std::vector<std::vector<int64_t>> create_strided_attention_pattern(
    int64_t seq_len, int64_t stride);

std::vector<std::vector<int64_t>> create_dilated_attention_pattern(
    int64_t seq_len, const std::vector<int64_t>& dilations);

std::vector<std::vector<int64_t>> create_bigbird_attention_pattern(
    int64_t seq_len, int64_t num_random_blocks, int64_t block_size);

std::vector<std::vector<int64_t>> create_longformer_attention_pattern(
    int64_t seq_len, int64_t window_size, 
    const std::vector<int64_t>& global_positions);

} // namespace patterns

} // namespace attention
} // namespace operators  
} // namespace deepcpp 