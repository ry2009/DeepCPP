#pragma once

#include "../../core/tensor/tensor.h"
#include "flash_attention.h"
#include <memory>
#include <vector>
#include <immintrin.h>
#include <cassert>

namespace deepcpp {
namespace operators {
namespace attention {

using deepcpp::core::Tensor;
using deepcpp::core::DataType;

/**
 * Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
 * 
 * MQA: Single key/value head shared across all query heads
 * GQA: Multiple query heads grouped to share fewer key/value heads
 * 
 * These methods significantly reduce memory usage and improve inference speed
 * while maintaining most of the model quality.
 */

/**
 * Multi-Query Attention
 * 
 * Uses a single key and value head across all query heads
 * Memory: O(seq_len * d_model) instead of O(seq_len * num_heads * head_dim)
 */
class MultiQueryAttention {
public:
    struct Config {
        int64_t num_query_heads;
        int64_t head_dim;
        int64_t block_size;          // For flash attention integration
        bool use_flash_attention;
        bool use_causal_mask;
        bool use_rotary_embeddings;
        float scale;
        float dropout_prob;
        
        Config(int64_t nqh = 32, int64_t hd = 64, int64_t bs = 64, 
               bool flash = true, bool causal = false, bool rope = false,
               float s = -1.0f, float dropout = 0.0f)
            : num_query_heads(nqh), head_dim(hd), block_size(bs),
              use_flash_attention(flash), use_causal_mask(causal),
              use_rotary_embeddings(rope), scale(s), dropout_prob(dropout) {}
    };
    
    MultiQueryAttention(const Config& config);
    
    /**
     * Forward pass with shared key/value heads
     * 
     * @param q Query tensor [batch, seq_len, num_query_heads, head_dim]
     * @param k Key tensor [batch, seq_len, 1, head_dim] - Single head
     * @param v Value tensor [batch, seq_len, 1, head_dim] - Single head
     * @param mask Optional attention mask
     * @return Output [batch, seq_len, num_query_heads, head_dim]
     */
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor& mask = Tensor());
    
    /**
     * Forward with KV cache for autoregressive generation
     */
    Tensor forward_with_cache(const Tensor& q, 
                             std::vector<Tensor>& k_cache,
                             std::vector<Tensor>& v_cache,
                             int64_t cache_position,
                             const Tensor& mask = Tensor());
    
    /**
     * Prefill phase for inference (encode all tokens at once)
     */
    Tensor prefill(const Tensor& q, const Tensor& k, const Tensor& v,
                   std::vector<Tensor>& k_cache, std::vector<Tensor>& v_cache,
                   const Tensor& mask = Tensor());
    
    /**
     * Decode phase for inference (generate one token)
     */
    Tensor decode(const Tensor& q, const Tensor& k, const Tensor& v,
                  std::vector<Tensor>& k_cache, std::vector<Tensor>& v_cache,
                  int64_t position, const Tensor& mask = Tensor());
    
    // Memory usage estimation
    size_t estimate_memory_usage(int64_t seq_len, int64_t batch_size = 1) const;
    size_t estimate_cache_memory(int64_t max_seq_len, int64_t batch_size = 1) const;
    
    // Performance analysis
    float get_memory_reduction_ratio() const;
    int64_t get_flops_reduction(int64_t seq_len) const;
    
private:
    Config config_;
    std::unique_ptr<FlashAttention> flash_attention_;
    
    void mqa_attention_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len,
        int64_t num_query_heads, int64_t head_dim,
        const float* mask_ptr = nullptr
    );
    
    void mqa_flash_attention_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len,
        int64_t num_query_heads, int64_t head_dim,
        const float* mask_ptr = nullptr
    );
    
    // Cache management
    void update_kv_cache(std::vector<Tensor>& k_cache, std::vector<Tensor>& v_cache,
                        const Tensor& new_k, const Tensor& new_v, int64_t position);
    
    // RoPE (Rotary Position Embeddings) support
    void apply_rotary_embeddings(Tensor& q, Tensor& k, int64_t position_offset = 0);
};

/**
 * Grouped-Query Attention
 * 
 * Groups multiple query heads to share fewer key/value heads
 * Generalizes MQA (group_size = num_query_heads) and MHA (group_size = 1)
 */
class GroupedQueryAttention {
public:
    struct Config {
        int64_t num_query_heads;
        int64_t num_kv_heads;        // Number of key/value head groups
        int64_t head_dim;
        int64_t block_size;
        bool use_flash_attention;
        bool use_causal_mask;
        bool use_rotary_embeddings;
        float scale;
        float dropout_prob;
        
        Config(int64_t nqh = 32, int64_t nkh = 8, int64_t hd = 64, int64_t bs = 64,
               bool flash = true, bool causal = false, bool rope = false,
               float s = -1.0f, float dropout = 0.0f)
            : num_query_heads(nqh), num_kv_heads(nkh), head_dim(hd), block_size(bs),
              use_flash_attention(flash), use_causal_mask(causal),
              use_rotary_embeddings(rope), scale(s), dropout_prob(dropout) {
            // Ensure query heads are evenly divisible by kv heads
            assert(num_query_heads % num_kv_heads == 0);
        }
    };
    
    GroupedQueryAttention(const Config& config);
    
    /**
     * Forward pass with grouped key/value heads
     * 
     * @param q Query tensor [batch, seq_len, num_query_heads, head_dim]
     * @param k Key tensor [batch, seq_len, num_kv_heads, head_dim]
     * @param v Value tensor [batch, seq_len, num_kv_heads, head_dim]
     * @param mask Optional attention mask
     * @return Output [batch, seq_len, num_query_heads, head_dim]
     */
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor& mask = Tensor());
    
    /**
     * Forward with KV cache for inference
     */
    Tensor forward_with_cache(const Tensor& q,
                             std::vector<Tensor>& k_cache,
                             std::vector<Tensor>& v_cache,
                             int64_t cache_position,
                             const Tensor& mask = Tensor());
    
    // Get group configuration
    int64_t get_group_size() const { return config_.num_query_heads / config_.num_kv_heads; }
    int64_t get_num_groups() const { return config_.num_kv_heads; }
    
    // Memory and performance analysis
    size_t estimate_memory_usage(int64_t seq_len, int64_t batch_size = 1) const;
    float get_memory_reduction_ratio() const;
    
private:
    Config config_;
    int64_t group_size_;             // Query heads per KV head group
    std::unique_ptr<FlashAttention> flash_attention_;
    
    void gqa_attention_kernel(
        const float* q_ptr, const float* k_ptr, const float* v_ptr,
        float* output_ptr, int64_t batch_size, int64_t seq_len,
        int64_t num_query_heads, int64_t num_kv_heads, int64_t head_dim,
        const float* mask_ptr = nullptr
    );
    
    // Expand KV heads to match query heads for computation
    void expand_kv_heads(const Tensor& kv_input, Tensor& expanded_output) const;
    
    // Apply rotary embeddings to grouped heads
    void apply_rotary_embeddings_grouped(Tensor& q, Tensor& k, int64_t position_offset = 0);
};

/**
 * Adaptive Multi-Query/Grouped-Query Attention
 * 
 * Dynamically chooses between MHA, GQA, and MQA based on:
 * - Available memory
 * - Sequence length
 * - Performance requirements
 */
class AdaptiveQueryAttention {
public:
    enum class AttentionMode {
        MULTI_HEAD,     // Standard MHA
        GROUPED_QUERY,  // GQA with configurable groups
        MULTI_QUERY     // MQA with single KV head
    };
    
    struct Config {
        int64_t num_query_heads;
        int64_t max_kv_heads;        // Maximum KV heads for GQA
        int64_t head_dim;
        
        // Adaptive thresholds
        size_t memory_threshold_mb;   // Switch to more efficient mode if memory exceeds
        int64_t seq_len_threshold;    // Switch based on sequence length
        
        // Performance targets
        float target_memory_reduction;
        float max_quality_loss;
        
        Config(int64_t nqh = 32, int64_t max_kh = 8, int64_t hd = 64,
               size_t mem_thresh = 1024, int64_t seq_thresh = 2048,
               float mem_reduction = 0.5f, float quality_loss = 0.05f)
            : num_query_heads(nqh), max_kv_heads(max_kh), head_dim(hd),
              memory_threshold_mb(mem_thresh), seq_len_threshold(seq_thresh),
              target_memory_reduction(mem_reduction), max_quality_loss(quality_loss) {}
    };
    
    AdaptiveQueryAttention(const Config& config);
    
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor& mask = Tensor());
    
    // Get current mode and statistics
    AttentionMode get_current_mode() const { return current_mode_; }
    int64_t get_current_kv_heads() const { return current_kv_heads_; }
    float get_current_memory_usage() const;
    
    // Manual mode control
    void set_mode(AttentionMode mode, int64_t kv_heads = 0);
    void enable_auto_mode() { auto_mode_ = true; }
    
private:
    Config config_;
    AttentionMode current_mode_;
    int64_t current_kv_heads_;
    bool auto_mode_;
    
    std::unique_ptr<GroupedQueryAttention> gqa_attention_;
    std::unique_ptr<MultiQueryAttention> mqa_attention_;
    
    AttentionMode select_optimal_mode(int64_t seq_len, int64_t batch_size) const;
    int64_t select_optimal_kv_heads(int64_t seq_len, int64_t batch_size) const;
    
    // Compress KV heads for mode switching
    Tensor compress_kv_heads(const Tensor& kv_input, int64_t target_heads) const;
};

/**
 * Paged Attention for Multi-Query/Grouped-Query
 * 
 * Specialized paged attention implementation optimized for MQA/GQA
 * with reduced memory footprint for KV cache
 */
class PagedQueryAttention {
public:
    struct PagedKVCache {
        std::vector<Tensor> key_pages;    // Reduced size for MQA/GQA
        std::vector<Tensor> value_pages;
        std::vector<int64_t> page_table;
        int64_t page_size;
        int64_t num_kv_heads;             // Reduced heads
        int64_t max_pages;
        int64_t current_length;
        
        PagedKVCache(int64_t ps, int64_t nkh, int64_t mp)
            : page_size(ps), num_kv_heads(nkh), max_pages(mp), current_length(0) {}
    };
    
    struct Config {
        int64_t page_size;
        int64_t max_pages;
        int64_t num_query_heads;
        int64_t num_kv_heads;
        int64_t head_dim;
        bool use_flash_attention;
        
        Config(int64_t ps = 64, int64_t mp = 1024, int64_t nqh = 32, 
               int64_t nkh = 8, int64_t hd = 64, bool flash = true)
            : page_size(ps), max_pages(mp), num_query_heads(nqh), 
              num_kv_heads(nkh), head_dim(hd), use_flash_attention(flash) {}
    };
    
    PagedQueryAttention(const Config& config);
    
    // Initialize cache with reduced KV heads
    void init_cache(PagedKVCache& cache, int64_t batch_size);
    
    // Append new tokens to cache
    void append_kv(PagedKVCache& cache, const Tensor& new_k, const Tensor& new_v);
    
    // Attention with paged cache
    Tensor forward_with_cache(const Tensor& q, PagedKVCache& cache,
                             const Tensor& mask = Tensor());
    
    // Memory management
    size_t get_cache_memory_usage(const PagedKVCache& cache) const;
    float get_memory_reduction_vs_mha() const;
    
private:
    Config config_;
    std::unique_ptr<GroupedQueryAttention> gqa_attention_;
    
    void allocate_new_page(PagedKVCache& cache);
    void copy_kv_to_page(PagedKVCache& cache, const Tensor& k, const Tensor& v, 
                         int64_t page_idx, int64_t offset);
    
    Tensor gather_from_pages(const PagedKVCache& cache, bool is_key) const;
};

// Utility functions
namespace mqa_utils {

/**
 * Memory analysis utilities
 */
size_t compute_mha_memory(int64_t seq_len, int64_t num_heads, int64_t head_dim);
size_t compute_mqa_memory(int64_t seq_len, int64_t num_query_heads, int64_t head_dim);
size_t compute_gqa_memory(int64_t seq_len, int64_t num_query_heads, 
                         int64_t num_kv_heads, int64_t head_dim);

float compute_memory_reduction_mqa(int64_t num_heads);
float compute_memory_reduction_gqa(int64_t num_query_heads, int64_t num_kv_heads);

/**
 * Performance analysis
 */
struct QueryAttentionBenchmark {
    std::string method_name;
    int64_t num_query_heads;
    int64_t num_kv_heads;
    float forward_time_ms;
    size_t memory_usage_mb;
    size_t cache_memory_mb;
    float memory_reduction_ratio;
    float throughput_tokens_per_sec;
};

std::vector<QueryAttentionBenchmark> benchmark_attention_variants(
    int64_t seq_len, int64_t batch_size, int64_t head_dim
);

/**
 * Quality analysis
 */
float compute_attention_quality_loss(const Tensor& mha_output, const Tensor& variant_output);

/**
 * KV head reduction strategies
 */
Tensor reduce_kv_heads_average(const Tensor& kv_input, int64_t target_heads);
Tensor reduce_kv_heads_learned(const Tensor& kv_input, int64_t target_heads, 
                              const Tensor& reduction_matrix);
Tensor reduce_kv_heads_clustering(const Tensor& kv_input, int64_t target_heads);

} // namespace mqa_utils

} // namespace attention
} // namespace operators
} // namespace deepcpp 