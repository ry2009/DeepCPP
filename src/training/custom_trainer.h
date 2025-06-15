#pragma once

#include "../core/tensor/tensor.h"
#include "../operators/attention/linear_attention.h"
#include "../operators/models/ssm.h"
#include "../operators/models/mixture_of_experts.h"
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <chrono>
#include <unordered_map>

namespace deepcpp {
namespace training {

using namespace core;
using namespace operators::attention;
using namespace operators::models;

/**
 * Training Data Structure
 */
struct TrainingBatch {
    Tensor input_ids;      // [batch_size, seq_len] - tokenized text
    Tensor attention_mask; // [batch_size, seq_len] - padding mask
    Tensor labels;         // [batch_size] - sentiment labels (0=negative, 1=positive)
    
    TrainingBatch(int batch_size, int seq_len)
        : input_ids({batch_size, seq_len}, DataType::INT32),
          attention_mask({batch_size, seq_len}, DataType::FLOAT32),
          labels({batch_size}, DataType::INT32) {}
};

/**
 * Simple Tokenizer for Text Processing
 */
class SimpleTokenizer {
public:
    SimpleTokenizer(int vocab_size = 10000, int max_seq_len = 512);
    
    std::vector<int> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int>& tokens);
    
    int get_vocab_size() const { return vocab_size_; }
    int get_max_seq_len() const { return max_seq_len_; }
    
    // Special tokens
    static constexpr int PAD_TOKEN = 0;
    static constexpr int UNK_TOKEN = 1;
    static constexpr int CLS_TOKEN = 2;
    static constexpr int SEP_TOKEN = 3;
    
private:
    int vocab_size_;
    int max_seq_len_;
    std::unordered_map<std::string, int> word_to_id_;
    std::unordered_map<int, std::string> id_to_word_;
    
    void build_vocabulary();
};

/**
 * Synthetic Data Generator for Sentiment Analysis
 */
class SentimentDataGenerator {
public:
    SentimentDataGenerator(std::shared_ptr<SimpleTokenizer> tokenizer);
    
    // Generate training data
    std::vector<TrainingBatch> generate_training_data(int num_batches, int batch_size);
    std::vector<TrainingBatch> generate_validation_data(int num_batches, int batch_size);
    
    // Load real data (if available)
    bool load_from_file(const std::string& filepath);
    
private:
    std::shared_ptr<SimpleTokenizer> tokenizer_;
    std::mt19937 rng_;
    
    // Predefined sentiment patterns
    std::vector<std::string> positive_words_;
    std::vector<std::string> negative_words_;
    std::vector<std::string> neutral_words_;
    
    std::string generate_sentiment_text(int label, int target_length);
    void initialize_word_lists();
};

/**
 * Custom Sentiment Analysis Model
 * Architecture: Embedding -> Linear Attention -> Mamba SSM -> MoE -> Classification Head
 */
class SentimentModel {
public:
    struct Config {
        int vocab_size = 10000;
        int d_model = 256;
        int max_seq_len = 512;
        int num_attention_heads = 8;
        int ssm_state_size = 64;
        int num_experts = 8;
        int expert_capacity = 2;
        int num_classes = 2;  // binary sentiment
        float dropout_prob = 0.1f;
        
        Config() = default;
    };
    
    SentimentModel(const Config& config);
    
    // Forward pass
    struct ModelOutput {
        Tensor logits;           // [batch_size, num_classes]
        Tensor hidden_states;    // [batch_size, seq_len, d_model]
        float moe_load_balance_loss = 0.0f;
        std::vector<float> expert_utilization;
    };
    
    ModelOutput forward(const Tensor& input_ids, const Tensor& attention_mask);
    
    // Training utilities
    float compute_loss(const ModelOutput& output, const Tensor& labels);
    std::vector<Tensor> get_parameters();
    void zero_gradients();
    void update_parameters(const std::vector<Tensor>& gradients, float learning_rate);
    
    // Model persistence
    void save_model(const std::string& filepath);
    void load_model(const std::string& filepath);
    
    const Config& get_config() const { return config_; }
    
private:
    Config config_;
    
    // Model components
    std::unique_ptr<Tensor> embedding_weights_;           // [vocab_size, d_model]
    std::unique_ptr<Tensor> position_embeddings_;         // [max_seq_len, d_model]
    std::unique_ptr<PerformerAttention> attention_layer_;
    std::unique_ptr<MambaSSM> ssm_layer_;
    std::unique_ptr<MixtureOfExperts> moe_layer_;
    std::unique_ptr<Tensor> classifier_weights_;          // [d_model, num_classes]
    std::unique_ptr<Tensor> classifier_bias_;             // [num_classes]
    
    // Layer normalization parameters
    std::unique_ptr<Tensor> ln1_weight_, ln1_bias_;       // Pre-attention
    std::unique_ptr<Tensor> ln2_weight_, ln2_bias_;       // Pre-SSM
    std::unique_ptr<Tensor> ln3_weight_, ln3_bias_;       // Pre-MoE
    std::unique_ptr<Tensor> ln_final_weight_, ln_final_bias_; // Pre-classifier
    
    void initialize_parameters();
    Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias);
    Tensor embedding_lookup(const Tensor& input_ids);
    Tensor add_position_embeddings(const Tensor& embeddings);
};

/**
 * Training Configuration
 */
struct TrainingConfig {
    int num_epochs = 10;
    int batch_size = 16;
    float learning_rate = 1e-4f;
    float weight_decay = 1e-5f;
    int warmup_steps = 100;
    int eval_steps = 500;
    int save_steps = 1000;
    bool use_mixed_precision = false;
    float gradient_clip_norm = 1.0f;
    std::string output_dir = "models/sentiment_model";
    
    TrainingConfig() = default;
};

/**
 * Custom Trainer Class
 */
class CustomTrainer {
public:
    CustomTrainer(std::shared_ptr<SentimentModel> model,
                  std::shared_ptr<SimpleTokenizer> tokenizer,
                  const TrainingConfig& config);
    
    // Main training loop
    void train();
    
    // Evaluation
    struct EvalResults {
        float accuracy = 0.0f;
        float loss = 0.0f;
        float f1_score = 0.0f;
        int correct_predictions = 0;
        int total_predictions = 0;
    };
    
    EvalResults evaluate(const std::vector<TrainingBatch>& eval_data);
    
    // Inference on single example
    struct PredictionResult {
        int predicted_label;
        float confidence;
        std::vector<float> class_probabilities;
    };
    
    PredictionResult predict(const std::string& text);
    
    // Training statistics
    struct TrainingStats {
        std::vector<float> train_losses;
        std::vector<float> eval_losses;
        std::vector<float> eval_accuracies;
        std::vector<float> learning_rates;
        int total_steps = 0;
        double total_training_time = 0.0;
    };
    
    const TrainingStats& get_training_stats() const { return stats_; }
    
private:
    std::shared_ptr<SentimentModel> model_;
    std::shared_ptr<SimpleTokenizer> tokenizer_;
    std::unique_ptr<SentimentDataGenerator> data_generator_;
    TrainingConfig config_;
    TrainingStats stats_;
    
    // Optimizer state
    std::vector<Tensor> momentum_buffers_;
    std::vector<Tensor> velocity_buffers_;  // For Adam optimizer
    int optimizer_step_ = 0;
    
    // Training utilities
    void initialize_optimizer();
    void optimizer_step(const std::vector<Tensor>& gradients);
    float get_learning_rate(int step);
    std::vector<Tensor> compute_gradients(const SentimentModel::ModelOutput& output, 
                                         const Tensor& labels);
    void clip_gradients(std::vector<Tensor>& gradients, float max_norm);
    
    // Logging and checkpointing
    void log_training_step(int step, float loss, float lr);
    void save_checkpoint(int step);
    void load_checkpoint(const std::string& checkpoint_path);
    
    // Metrics computation
    float compute_accuracy(const Tensor& predictions, const Tensor& labels);
    float compute_f1_score(const Tensor& predictions, const Tensor& labels);
    Tensor softmax(const Tensor& logits);
};

} // namespace training
} // namespace deepcpp 