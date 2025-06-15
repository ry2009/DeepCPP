#include "mixture_of_experts.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace deepcpp {
namespace operators {
namespace models {

// FeedforwardExpert Implementation
FeedforwardExpert::FeedforwardExpert(const Config& config, int64_t expert_id) 
    : config_(config), expert_id_(expert_id) {
    initialize_weights();
}

void FeedforwardExpert::initialize_weights() {
    // Initialize W1 (input to hidden)
    std::vector<int64_t> w1_shape = {config_.d_model, config_.d_ff};
    w1_ = std::make_unique<Tensor>(w1_shape, core::DataType::FLOAT32);
    
    // Initialize W2 (hidden to output)  
    std::vector<int64_t> w2_shape = {config_.d_ff, config_.d_model};
    w2_ = std::make_unique<Tensor>(w2_shape, core::DataType::FLOAT32);
    
    // Initialize biases
    std::vector<int64_t> b1_shape = {config_.d_ff};
    std::vector<int64_t> b2_shape = {config_.d_model};
    b1_ = std::make_unique<Tensor>(b1_shape, core::DataType::FLOAT32);
    b2_ = std::make_unique<Tensor>(b2_shape, core::DataType::FLOAT32);
    
    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    float w1_std = std::sqrt(2.0f / (config_.d_model + config_.d_ff));
    float w2_std = std::sqrt(2.0f / (config_.d_ff + config_.d_model));
    
    std::normal_distribution<float> w1_dist(0.0f, w1_std);
    std::normal_distribution<float> w2_dist(0.0f, w2_std);
    
    // Initialize W1
    float* w1_data = w1_->data_ptr<float>();
    for (int64_t i = 0; i < w1_->numel(); ++i) {
        w1_data[i] = w1_dist(gen);
    }
    
    // Initialize W2
    float* w2_data = w2_->data_ptr<float>();
    for (int64_t i = 0; i < w2_->numel(); ++i) {
        w2_data[i] = w2_dist(gen);
    }
    
    // Initialize biases to zero
    std::fill(b1_->data_ptr<float>(), b1_->data_ptr<float>() + b1_->numel(), 0.0f);
    std::fill(b2_->data_ptr<float>(), b2_->data_ptr<float>() + b2_->numel(), 0.0f);
    
    call_count_ = 0;
    total_tokens_ = 0;
}

Tensor FeedforwardExpert::forward(const Tensor& input) {
    const auto& input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t seq_len = input_shape[1];
    int64_t d_model = input_shape[2];
    
    call_count_++;
    total_tokens_ += batch_size * seq_len;
    
    // Reshape input for matrix multiplication: [batch*seq, d_model]
    int64_t total_tokens = batch_size * seq_len;
    
    // Hidden layer: input @ W1 + b1
    std::vector<int64_t> hidden_shape = {batch_size, seq_len, config_.d_ff};
    Tensor hidden(hidden_shape, input.dtype());
    
    // Matrix multiplication: input @ W1
    const float* input_data = input.data_ptr<float>();
    const float* w1_data = w1_->data_ptr<float>();
    const float* b1_data = b1_->data_ptr<float>();
    float* hidden_data = hidden.data_ptr<float>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            for (int64_t h = 0; h < config_.d_ff; ++h) {
                float sum = b1_data[h]; // bias
                for (int64_t d = 0; d < d_model; ++d) {
                    int64_t input_idx = b * seq_len * d_model + s * d_model + d;
                    int64_t w1_idx = d * config_.d_ff + h;
                    sum += input_data[input_idx] * w1_data[w1_idx];
                }
                int64_t hidden_idx = b * seq_len * config_.d_ff + s * config_.d_ff + h;
                
                // Apply activation function
                if (config_.activation == "gelu") {
                    hidden_data[hidden_idx] = gelu_activation(sum);
                } else if (config_.activation == "swiglu") {
                    // For SwiGLU, we need gate mechanism (simplified here)
                    hidden_data[hidden_idx] = sum * sigmoid_activation(sum);
                } else { // relu
                    hidden_data[hidden_idx] = std::max(0.0f, sum);
                }
            }
        }
    }
    
    // Output layer: hidden @ W2 + b2
    Tensor output(input_shape, input.dtype());
    const float* w2_data = w2_->data_ptr<float>();
    const float* b2_data = b2_->data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            for (int64_t d = 0; d < d_model; ++d) {
                float sum = b2_data[d]; // bias
                for (int64_t h = 0; h < config_.d_ff; ++h) {
                    int64_t hidden_idx = b * seq_len * config_.d_ff + s * config_.d_ff + h;
                    int64_t w2_idx = h * d_model + d;
                    sum += hidden_data[hidden_idx] * w2_data[w2_idx];
                }
                int64_t output_idx = b * seq_len * d_model + s * d_model + d;
                output_data[output_idx] = sum;
            }
        }
    }
    
    return output;
}

float FeedforwardExpert::gelu_activation(float x) const {
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

float FeedforwardExpert::sigmoid_activation(float x) const {
    return 1.0f / (1.0f + std::exp(-x));
}

size_t FeedforwardExpert::get_parameter_count() const { 
    return w1_->numel() + w2_->numel() + b1_->numel() + b2_->numel();
}

size_t FeedforwardExpert::estimate_memory_usage(int64_t batch_size, int64_t seq_len) const { 
    return get_parameter_count() * sizeof(float) + 
           batch_size * seq_len * config_.d_ff * sizeof(float); // hidden activations
}

void FeedforwardExpert::reset_statistics() {
    call_count_ = 0;
    total_tokens_ = 0;
}

float FeedforwardExpert::get_load() const { 
    return static_cast<float>(total_tokens_);
}

// RouterNetwork Implementation
RouterNetwork::RouterNetwork(const Config& config, int64_t d_model) 
    : config_(config), d_model_(d_model) {
    initialize_routing_weights();
}

void RouterNetwork::initialize_routing_weights() {
    // Router weight matrix: [d_model, num_experts]
    std::vector<int64_t> router_shape = {d_model_, config_.num_experts};
    router_weights_ = std::make_unique<Tensor>(router_shape, core::DataType::FLOAT32);
    
    // Initialize with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    float std_dev = std::sqrt(2.0f / d_model_);
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    float* router_data = router_weights_->data_ptr<float>();
    for (int64_t i = 0; i < router_weights_->numel(); ++i) {
        router_data[i] = dist(gen);
    }
    
    expert_counts_.resize(config_.num_experts, 0);
    total_tokens_ = 0;
}

RouterNetwork::RoutingResult RouterNetwork::route(const Tensor& input) {
    const auto& input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t seq_len = input_shape[1];
    int64_t d_model = input_shape[2];
    
    total_tokens_ += batch_size * seq_len;
    
    RoutingResult result;
    
    // Compute router logits: input @ router_weights
    std::vector<int64_t> logits_shape = {batch_size, seq_len, config_.num_experts};
    result.logits = std::make_unique<Tensor>(logits_shape, input.dtype());
    
    const float* input_data = input.data_ptr<float>();
    const float* router_data = router_weights_->data_ptr<float>();
    float* logits_data = result.logits->data_ptr<float>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            for (int64_t e = 0; e < config_.num_experts; ++e) {
                float sum = 0.0f;
                for (int64_t d = 0; d < d_model; ++d) {
                    int64_t input_idx = b * seq_len * d_model + s * d_model + d;
                    int64_t router_idx = d * config_.num_experts + e;
                    sum += input_data[input_idx] * router_data[router_idx];
                }
                int64_t logits_idx = b * seq_len * config_.num_experts + s * config_.num_experts + e;
                logits_data[logits_idx] = sum;
            }
        }
    }
    
    // Apply softmax and top-k selection
    std::vector<int64_t> indices_shape = {batch_size, seq_len, config_.top_k};
    std::vector<int64_t> weights_shape = {batch_size, seq_len, config_.top_k};
    
    result.expert_indices = std::make_unique<Tensor>(indices_shape, core::DataType::INT64);
    result.expert_weights = std::make_unique<Tensor>(weights_shape, input.dtype());
    
    int64_t* indices_data = result.expert_indices->data_ptr<int64_t>();
    float* weights_data = result.expert_weights->data_ptr<float>();
    
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            // Get logits for this token
            std::vector<std::pair<float, int64_t>> expert_scores;
            for (int64_t e = 0; e < config_.num_experts; ++e) {
                int64_t logits_idx = b * seq_len * config_.num_experts + s * config_.num_experts + e;
                expert_scores.push_back({logits_data[logits_idx], e});
            }
            
            // Sort by score (descending)
            std::sort(expert_scores.begin(), expert_scores.end(), 
                     [](const auto& a, const auto& b) { return a.first > b.first; });
            
            // Apply softmax to top-k
            std::vector<float> top_k_logits;
            for (int64_t k = 0; k < config_.top_k; ++k) {
                top_k_logits.push_back(expert_scores[k].first);
            }
            
            // Softmax
            float max_logit = *std::max_element(top_k_logits.begin(), top_k_logits.end());
            float sum_exp = 0.0f;
            for (float& logit : top_k_logits) {
                logit = std::exp(logit - max_logit);
                sum_exp += logit;
            }
            
            // Store results
            for (int64_t k = 0; k < config_.top_k; ++k) {
                int64_t result_idx = b * seq_len * config_.top_k + s * config_.top_k + k;
                indices_data[result_idx] = expert_scores[k].second;
                weights_data[result_idx] = top_k_logits[k] / sum_exp;
                
                // Update expert usage statistics
                expert_counts_[expert_scores[k].second]++;
            }
        }
    }
    
    // Compute load balancing loss
    result.load_balancing_loss = compute_load_balancing_loss(*result.logits);
    
    return result;
}

float RouterNetwork::compute_load_balancing_loss(const Tensor& logits) const {
    // Simplified load balancing loss: encourage uniform expert usage
    const float* logits_data = logits.data_ptr<float>();
    const auto& shape = logits.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];
    int64_t num_experts = shape[2];
    
    std::vector<float> expert_probs(num_experts, 0.0f);
    int64_t total_tokens = batch_size * seq_len;
    
    // Compute average probability for each expert
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            // Softmax over experts for this token
            std::vector<float> token_logits(num_experts);
            float max_logit = -std::numeric_limits<float>::infinity();
            
            for (int64_t e = 0; e < num_experts; ++e) {
                int64_t idx = b * seq_len * num_experts + s * num_experts + e;
                token_logits[e] = logits_data[idx];
                max_logit = std::max(max_logit, token_logits[e]);
            }
            
            float sum_exp = 0.0f;
            for (int64_t e = 0; e < num_experts; ++e) {
                token_logits[e] = std::exp(token_logits[e] - max_logit);
                sum_exp += token_logits[e];
            }
            
            for (int64_t e = 0; e < num_experts; ++e) {
                expert_probs[e] += token_logits[e] / sum_exp;
            }
        }
    }
    
    // Normalize by total tokens
    for (float& prob : expert_probs) {
        prob /= total_tokens;
    }
    
    // Compute coefficient of variation as load balancing loss
    float mean_prob = 1.0f / num_experts;
    float variance = 0.0f;
    for (float prob : expert_probs) {
        variance += (prob - mean_prob) * (prob - mean_prob);
    }
    variance /= num_experts;
    
    return std::sqrt(variance) / mean_prob; // Coefficient of variation
}

std::vector<float> RouterNetwork::get_expert_utilization() const { 
    std::vector<float> utilization(config_.num_experts);
    if (total_tokens_ > 0) {
        for (size_t i = 0; i < expert_counts_.size(); ++i) {
            utilization[i] = static_cast<float>(expert_counts_[i]) / total_tokens_;
        }
    }
    return utilization;
}

float RouterNetwork::get_routing_entropy() const { 
    auto utilization = get_expert_utilization();
    float entropy = 0.0f;
    for (float p : utilization) {
        if (p > 0) {
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

void RouterNetwork::reset_statistics() {
    std::fill(expert_counts_.begin(), expert_counts_.end(), 0);
    total_tokens_ = 0;
}

// MixtureOfExperts Implementation
MixtureOfExperts::MixtureOfExperts(const Config& config) : config_(config) {
    // Create router with RouterNetwork::Config
    RouterNetwork::Config router_config;
    router_config.num_experts = config.num_experts;
    router_config.top_k = 2; // Default top-k
    router_ = std::make_unique<RouterNetwork>(router_config, config.d_model);
    
    // Create experts
    FeedforwardExpert::Config expert_config;
    expert_config.d_model = config.d_model;
    expert_config.d_ff = config.d_ff;
    
    for (int64_t i = 0; i < config.num_experts; ++i) {
        auto expert = std::make_unique<FeedforwardExpert>(expert_config, i);
        experts_.push_back(std::move(expert));
    }
}

MixtureOfExperts::MoEOutput MixtureOfExperts::forward(const Tensor& input) {
    MoEOutput output;
    
    // Route tokens to experts
    auto routing_result = router_->route(input);
    
    // Initialize output tensor
    output.output = Tensor(input.shape(), input.dtype());
    std::fill(output.output.data_ptr<float>(), 
              output.output.data_ptr<float>() + output.output.numel(), 0.0f);
    
    const auto& input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t seq_len = input_shape[1];
    int64_t d_model = input_shape[2];
    
    const int64_t* indices_data = routing_result.expert_indices->data_ptr<int64_t>();
    const float* weights_data = routing_result.expert_weights->data_ptr<float>();
    const float* input_data = input.data_ptr<float>();
    float* output_data = output.output.data_ptr<float>();
    
    int64_t top_k = 2; // Use default top-k value
    
    // Process each token
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            
            // Create input tensor for this token
            std::vector<int64_t> token_shape = {1, 1, d_model};
            Tensor token_input(token_shape, input.dtype());
            float* token_input_data = token_input.data_ptr<float>();
            
            for (int64_t d = 0; d < d_model; ++d) {
                int64_t input_idx = b * seq_len * d_model + s * d_model + d;
                token_input_data[d] = input_data[input_idx];
            }
            
            // Process with top-k experts
            for (int64_t k = 0; k < top_k; ++k) {
                int64_t routing_idx = b * seq_len * top_k + s * top_k + k;
                int64_t expert_id = indices_data[routing_idx];
                float weight = weights_data[routing_idx];
                
                // Forward through expert
                auto expert_output = experts_[expert_id]->forward(token_input);
                const float* expert_data = expert_output.data_ptr<float>();
                
                // Add weighted expert output to final output
                for (int64_t d = 0; d < d_model; ++d) {
                    int64_t output_idx = b * seq_len * d_model + s * d_model + d;
                    output_data[output_idx] += weight * expert_data[d];
                }
            }
        }
    }
    
    output.load_balance_loss = routing_result.load_balancing_loss;
    output.expert_utilization = router_->get_expert_utilization();
    output.routing_entropy = router_->get_routing_entropy();
    
    return output;
}

void MixtureOfExperts::add_expert(std::unique_ptr<ExpertBase> expert) {
    experts_.push_back(std::move(expert));
    config_.num_experts = experts_.size();
}

void MixtureOfExperts::remove_expert(int64_t expert_id) {
    if (expert_id >= 0 && expert_id < static_cast<int64_t>(experts_.size())) {
        experts_.erase(experts_.begin() + expert_id);
        config_.num_experts = experts_.size();
    }
}

void MixtureOfExperts::freeze_expert(int64_t expert_id, bool freeze) {
    // Implementation would set expert parameters as non-trainable
    // For now, just mark in config
    (void)expert_id; (void)freeze; // Suppress warnings
}

std::vector<float> MixtureOfExperts::get_expert_loads() const { 
    std::vector<float> loads;
    for (const auto& expert : experts_) {
        loads.push_back(expert->get_load());
    }
    return loads;
}

void MixtureOfExperts::print_routing_statistics() const {
    auto utilization = router_->get_expert_utilization();
    std::cout << "Expert Utilization:\n";
    for (size_t i = 0; i < utilization.size(); ++i) {
        std::cout << "  Expert " << i << ": " << utilization[i] * 100.0f << "%\n";
    }
    std::cout << "Routing Entropy: " << router_->get_routing_entropy() << "\n";
}

void MixtureOfExperts::reset_all_statistics() {
    router_->reset_statistics();
    for (auto& expert : experts_) {
        expert->reset_statistics();
    }
}

size_t MixtureOfExperts::estimate_memory_usage(int64_t batch_size, int64_t seq_len) const { 
    size_t total = 0;
    for (const auto& expert : experts_) {
        total += expert->estimate_memory_usage(batch_size, seq_len);
    }
    // Add router memory (simplified)
    total += config_.d_model * config_.num_experts * sizeof(float);
    // Add routing result memory
    total += batch_size * seq_len * config_.num_experts * sizeof(float); // logits
    total += batch_size * seq_len * 2 * (sizeof(int64_t) + sizeof(float)); // indices + weights (top-k=2)
    return total;
}

float MixtureOfExperts::get_model_efficiency() const { 
    auto utilization = router_->get_expert_utilization();
    float active_experts = 0.0f;
    for (float u : utilization) {
        if (u > 0.01f) active_experts += 1.0f; // Count experts with >1% utilization
    }
    return active_experts / config_.num_experts;
}

} // namespace models
} // namespace operators
} // namespace deepcpp 