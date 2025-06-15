#pragma once

#include "../../core/tensor/tensor.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <immintrin.h>

namespace deepcpp {
namespace operators {
namespace models {

using deepcpp::core::Tensor;
using deepcpp::core::DataType;

/**
 * Mixture of Experts (MoE) Implementation
 * 
 * Supports various MoE architectures:
 * - Switch Transformer MoE
 * - GLaM (Generalist Language Model) 
 * - PaLM MoE
 * - Expert Choice routing
 * - Hash routing
 * - Adaptive routing
 */

enum class RoutingStrategy {
    TOP_K,              // Standard top-k routing
    SWITCH,             // Switch Transformer (top-1 with load balancing)
    EXPERT_CHOICE,      // Expert Choice routing
    HASH_ROUTING,       // Hash-based routing
    LEARNED_ROUTING,    // Learned routing with RL
    ADAPTIVE_ROUTING,   // Adaptive capacity routing
    SOFT_ROUTING,       // Soft routing (all experts weighted)
    BALANCED_ROUTING    // Balanced assignment with constraints
};

enum class ExpertType {
    FEEDFORWARD,        // Standard FFN experts
    ATTENTION,          // Attention-based experts  
    CONVOLUTION,        // Convolutional experts
    RECURRENT,          // RNN/LSTM experts
    TRANSFORMER_BLOCK,  // Full transformer block experts
    CUSTOM              // Custom expert implementation
};

/**
 * Base Expert Interface
 */
class ExpertBase {
public:
    virtual ~ExpertBase() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual size_t get_parameter_count() const = 0;
    virtual size_t estimate_memory_usage(int64_t batch_size, int64_t seq_len) const = 0;
    virtual void reset_statistics() = 0;
    virtual float get_load() const = 0;  // Current utilization
    
protected:
    mutable int64_t forward_count_ = 0;
    mutable float total_load_ = 0.0f;
};

/**
 * Feedforward Expert
 * 
 * Standard FFN expert with configurable architecture
 */
class FeedforwardExpert : public ExpertBase {
public:
    struct Config {
        int64_t d_model;
        int64_t d_ff;
        std::string activation;     // "relu", "gelu", "swish", "geglu"
        float dropout_prob;
        bool use_bias;
        bool use_layer_norm;
        
        Config(int64_t dm = 512, int64_t dff = 2048, const std::string& act = "gelu",
               float dropout = 0.1f, bool bias = true, bool ln = false)
            : d_model(dm), d_ff(dff), activation(act), dropout_prob(dropout),
              use_bias(bias), use_layer_norm(ln) {}
    };
    
    FeedforwardExpert(const Config& config, int64_t expert_id);
    
    Tensor forward(const Tensor& input) override;
    size_t get_parameter_count() const override;
    size_t estimate_memory_usage(int64_t batch_size, int64_t seq_len) const override;
    void reset_statistics() override;
    float get_load() const override;
    
private:
    Config config_;
    int64_t expert_id_;
    
    // Parameters
    std::unique_ptr<Tensor> w1_;          // First linear layer
    std::unique_ptr<Tensor> b1_;          // First bias
    std::unique_ptr<Tensor> w2_;          // Second linear layer  
    std::unique_ptr<Tensor> b2_;          // Second bias
    std::unique_ptr<Tensor> gate_w_;      // For GLU variants
    
    // Layer norm (optional)
    std::unique_ptr<Tensor> ln_weight_;
    std::unique_ptr<Tensor> ln_bias_;
    
    void initialize_weights();
    float gelu_activation(float x) const;
    float sigmoid_activation(float x) const;
    
    // Statistics
    mutable int64_t call_count_ = 0;
    mutable int64_t total_tokens_ = 0;
};

/**
 * Gating/Router Network
 * 
 * Routes tokens to appropriate experts based on various strategies
 */
class RouterNetwork {
public:
    struct Config {
        RoutingStrategy strategy;
        int64_t num_experts;
        int64_t top_k;                    // For top-k routing
        int64_t expert_capacity;          // Maximum tokens per expert
        float load_balance_loss_weight;   // Weight for load balancing loss
        float router_z_loss_weight;       // Weight for router z-loss
        bool use_auxiliary_loss;          // Use auxiliary load balancing loss
        bool jitter_noise;                // Add jitter noise for training
        float noise_epsilon;              // Noise strength
        
        Config(RoutingStrategy rs = RoutingStrategy::TOP_K, int64_t ne = 8, int64_t k = 2,
               int64_t cap = 0, float lb_weight = 0.01f, float z_weight = 0.001f,
               bool aux_loss = true, bool jitter = true, float noise = 0.1f)
            : strategy(rs), num_experts(ne), top_k(k), expert_capacity(cap),
              load_balance_loss_weight(lb_weight), router_z_loss_weight(z_weight),
              use_auxiliary_loss(aux_loss), jitter_noise(jitter), noise_epsilon(noise) {}
    };
    
    RouterNetwork(const Config& config, int64_t d_model);
    
    struct RoutingResult {
        std::unique_ptr<Tensor> logits;           // [batch, seq_len, num_experts] - router logits
        std::unique_ptr<Tensor> expert_indices;   // [batch, seq_len, top_k] - selected expert indices
        std::unique_ptr<Tensor> expert_weights;   // [batch, seq_len, top_k] - routing weights
        float load_balancing_loss;                // Auxiliary load balancing loss
    };
    
    RoutingResult route(const Tensor& input);
    
    // Get routing statistics
    std::vector<float> get_expert_utilization() const;
    float get_routing_entropy() const;
    void reset_statistics();
    
private:
    Config config_;
    int64_t d_model_;
    
    // Router parameters
    std::unique_ptr<Tensor> router_weights_;    // [d_model, num_experts]
    std::unique_ptr<Tensor> router_bias_;       // [num_experts]
    
    // Statistics tracking
    mutable std::vector<int64_t> expert_counts_;
    mutable int64_t total_tokens_;
    
    void initialize_routing_weights();
    float compute_load_balancing_loss(const Tensor& logits) const;
    
    RoutingResult top_k_routing(const Tensor& logits, const Tensor& input);
    RoutingResult switch_routing(const Tensor& logits, const Tensor& input);
    RoutingResult expert_choice_routing(const Tensor& logits, const Tensor& input);
    RoutingResult hash_routing(const Tensor& input);
    RoutingResult soft_routing(const Tensor& logits, const Tensor& input);
    
    float compute_load_balance_loss(const Tensor& routing_weights, 
                                   const std::vector<int64_t>& expert_counts,
                                   int64_t total_tokens);
    float compute_router_z_loss(const Tensor& logits);
    
    void apply_capacity_constraints(RoutingResult& result);
    void add_jitter_noise(Tensor& logits);
};

/**
 * Main Mixture of Experts Layer
 */
class MixtureOfExperts {
public:
    struct Config {
        ExpertType expert_type;
        int64_t num_experts;
        int64_t d_model;
        int64_t d_ff;                     // For FFN experts
        int64_t top_k;                    // Top-k routing
        RouterNetwork::Config router_config;
        FeedforwardExpert::Config expert_config;
        bool enable_expert_parallelism;   // Parallelize expert computation
        bool use_expert_dropout;          // Dropout entire experts during training
        float expert_dropout_prob;
        
        Config(ExpertType et = ExpertType::FEEDFORWARD, int64_t ne = 8, 
               int64_t dm = 512, int64_t dff = 2048, int64_t k = 2,
               bool parallel = false, bool exp_dropout = false, float dropout_prob = 0.1f)
            : expert_type(et), num_experts(ne), d_model(dm), d_ff(dff), top_k(k),
              enable_expert_parallelism(parallel), use_expert_dropout(exp_dropout),
              expert_dropout_prob(dropout_prob) {}
    };
    
    MixtureOfExperts(const Config& config);
    
    struct MoEOutput {
        Tensor output;              // Main output tensor
        float load_balance_loss;    // Auxiliary load balancing loss
        float router_z_loss;        // Router z-loss
        std::vector<float> expert_utilization;  // Per-expert utilization
        int64_t active_experts;     // Number of experts used
        float routing_entropy;      // Routing diversity measure
    };
    
    MoEOutput forward(const Tensor& input);
    
    // Expert management
    void add_expert(std::unique_ptr<ExpertBase> expert);
    void remove_expert(int64_t expert_id);
    void freeze_expert(int64_t expert_id, bool freeze = true);
    
    // Statistics and analysis
    std::vector<float> get_expert_loads() const;
    void print_routing_statistics() const;
    void reset_all_statistics();
    
    // Memory and performance
    size_t estimate_memory_usage(int64_t batch_size, int64_t seq_len) const;
    float get_model_efficiency() const;  // Activated parameters / total parameters
    
private:
    Config config_;
    std::unique_ptr<RouterNetwork> router_;
    std::vector<std::unique_ptr<ExpertBase>> experts_;
    std::vector<bool> expert_frozen_;
    
    MoEOutput dispatch_and_combine(const Tensor& input, 
                                  const RouterNetwork::RoutingResult& routing);
    
    // Efficient sparse computation
    Tensor sparse_expert_computation(const Tensor& input,
                                   const Tensor& expert_indices,
                                   const Tensor& routing_weights,
                                   const Tensor& expert_mask);
    
    // Parallel expert execution
    std::vector<Tensor> parallel_expert_forward(const std::vector<Tensor>& expert_inputs);
    
    // Expert dropout during training
    void apply_expert_dropout(std::vector<bool>& active_experts);
};

/**
 * Hierarchical Mixture of Experts
 * 
 * Multi-level MoE with coarse and fine-grained expert selection
 */
class HierarchicalMoE {
public:
    struct Config {
        int64_t num_coarse_experts;     // First level experts
        int64_t num_fine_experts;       // Second level experts per coarse expert
        int64_t d_model;
        MixtureOfExperts::Config coarse_config;
        MixtureOfExperts::Config fine_config;
        bool share_fine_experts;        // Share fine experts across coarse experts
        
        Config(int64_t nce = 4, int64_t nfe = 4, int64_t dm = 512, bool share = false)
            : num_coarse_experts(nce), num_fine_experts(nfe), d_model(dm), 
              share_fine_experts(share) {}
    };
    
    HierarchicalMoE(const Config& config);
    
    MixtureOfExperts::MoEOutput forward(const Tensor& input);
    
    size_t estimate_memory_usage(int64_t batch_size, int64_t seq_len) const;
    
private:
    Config config_;
    std::unique_ptr<MixtureOfExperts> coarse_moe_;
    std::vector<std::unique_ptr<MixtureOfExperts>> fine_moes_;
};

/**
 * Adaptive Mixture of Experts
 * 
 * Dynamically adjusts number of active experts based on:
 * - Input complexity
 * - Available computational budget
 * - Performance requirements
 */
class AdaptiveMoE {
public:
    struct Config {
        int64_t min_experts;            // Minimum active experts
        int64_t max_experts;            // Maximum active experts  
        float complexity_threshold;     // Threshold for expert activation
        float budget_constraint;        // Computational budget limit
        bool use_early_exit;           // Early exit based on confidence
        float early_exit_threshold;
        
        Config(int64_t min_exp = 1, int64_t max_exp = 8, float comp_thresh = 0.5f,
               float budget = 1.0f, bool early_exit = false, float exit_thresh = 0.9f)
            : min_experts(min_exp), max_experts(max_exp), complexity_threshold(comp_thresh),
              budget_constraint(budget), use_early_exit(early_exit), 
              early_exit_threshold(exit_thresh) {}
    };
    
    AdaptiveMoE(const Config& config, int64_t d_model);
    
    MixtureOfExperts::MoEOutput forward(const Tensor& input);
    
    // Adaptation controls
    void set_budget_constraint(float budget) { config_.budget_constraint = budget; }
    void set_complexity_threshold(float threshold) { config_.complexity_threshold = threshold; }
    
    // Performance monitoring
    float get_average_experts_used() const;
    float get_computational_savings() const;
    
private:
    Config config_;
    std::unique_ptr<MixtureOfExperts> base_moe_;
    
    // Complexity estimation network
    std::unique_ptr<Tensor> complexity_weights_;
    std::unique_ptr<Tensor> complexity_bias_;
    
    // Statistics
    mutable float total_experts_used_;
    mutable int64_t forward_calls_;
    
    float estimate_input_complexity(const Tensor& input);
    int64_t select_num_experts(float complexity, float budget);
    float compute_early_exit_confidence(const Tensor& intermediate_output);
};

/**
 * MoE Training Utilities
 */
class MoETrainingManager {
public:
    struct TrainingConfig {
        float expert_dropout_prob;      // Probability of dropping experts
        float load_balance_weight;      // Weight for load balancing loss
        float diversity_weight;         // Weight for expert diversity loss
        bool use_gradual_unfreezing;    // Gradually unfreeze experts
        int64_t unfreezing_schedule;    // Steps between expert unfreezing
        bool use_expert_regularization; // L2 regularization on expert weights
        float expert_reg_weight;
        
        TrainingConfig(float exp_dropout = 0.1f, float lb_weight = 0.01f,
                      float div_weight = 0.001f, bool grad_unfreeze = false,
                      int64_t unfreeze_steps = 1000, bool exp_reg = false,
                      float reg_weight = 0.0001f)
            : expert_dropout_prob(exp_dropout), load_balance_weight(lb_weight),
              diversity_weight(div_weight), use_gradual_unfreezing(grad_unfreeze),
              unfreezing_schedule(unfreeze_steps), use_expert_regularization(exp_reg),
              expert_reg_weight(reg_weight) {}
    };
    
    MoETrainingManager(const TrainingConfig& config);
    
    // Compute auxiliary losses
    float compute_total_auxiliary_loss(const MixtureOfExperts::MoEOutput& moe_output);
    float compute_expert_diversity_loss(const std::vector<std::unique_ptr<ExpertBase>>& experts);
    
    // Training schedule management
    void step(int64_t current_step);
    bool should_unfreeze_expert(int64_t expert_id, int64_t current_step) const;
    
    // Expert initialization strategies
    void initialize_expert_gradually(MixtureOfExperts& moe, int64_t current_step);
    void apply_expert_regularization(const std::vector<std::unique_ptr<ExpertBase>>& experts);
    
private:
    TrainingConfig config_;
    std::vector<int64_t> expert_unfreeze_steps_;
    int64_t current_training_step_;
};

// Utility functions for MoE
namespace moe_utils {

/**
 * Expert analysis utilities
 */
float compute_expert_specialization(const std::vector<std::unique_ptr<ExpertBase>>& experts);
float compute_routing_balance(const std::vector<int64_t>& expert_counts);
std::vector<float> analyze_expert_gradients(const std::vector<std::unique_ptr<ExpertBase>>& experts);

/**
 * Load balancing utilities
 */
float compute_coefficient_of_variation(const std::vector<int64_t>& counts);
float compute_entropy(const std::vector<float>& probabilities);
std::vector<float> compute_load_balance_weights(const std::vector<int64_t>& expert_counts);

/**
 * Memory optimization
 */
size_t compute_moe_memory_footprint(int64_t num_experts, int64_t d_model, 
                                   int64_t d_ff, int64_t batch_size, int64_t seq_len);
float compute_memory_efficiency(const MixtureOfExperts& moe, int64_t active_experts);

/**
 * Performance benchmarking
 */
struct MoEBenchmark {
    std::string config_name;
    int64_t num_experts;
    int64_t active_experts;
    float forward_time_ms;
    size_t memory_usage_mb;
    float expert_utilization;
    float load_balance_score;
    float throughput_tokens_per_sec;
};

std::vector<MoEBenchmark> benchmark_moe_configurations(
    int64_t seq_len, int64_t d_model, int64_t batch_size
);

/**
 * Routing strategy analysis
 */
struct RoutingAnalysis {
    RoutingStrategy strategy;
    float load_balance_score;
    float routing_entropy;
    float expert_utilization;
    int64_t active_experts;
    float computational_efficiency;
};

std::vector<RoutingAnalysis> analyze_routing_strategies(
    const Tensor& input, const std::vector<RoutingStrategy>& strategies
);

} // namespace moe_utils

} // namespace models
} // namespace operators
} // namespace deepcpp 