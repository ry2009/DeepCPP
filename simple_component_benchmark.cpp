#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <random>
#include <iomanip>
#include <numeric>
#include <algorithm>

// Include our core components
#include "src/core/tensor/tensor.h"
#include "src/operators/attention/sparse_attention.h"
#include "src/operators/attention/linear_attention.h"
#include "src/operators/attention/multi_query_attention.h"
#include "src/operators/models/ssm.h"
#include "src/operators/models/mixture_of_experts.h"
#include "src/operators/performance/simd_kernels.h"

using namespace deepcpp;

class SimpleTimer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    
public:
    SimpleTimer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0;
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

std::shared_ptr<core::Tensor> create_random_tensor(const std::vector<int64_t>& shape) {
    auto tensor = std::make_shared<core::Tensor>(shape, core::DataType::FLOAT32);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    float* data = static_cast<float*>(tensor->data_ptr());
    size_t total_size = tensor->numel();
    
    for (size_t i = 0; i < total_size; ++i) {
        data[i] = dis(gen);
    }
    
    return tensor;
}

void benchmark_sparse_attention() {
    std::cout << "=== Benchmarking Sparse Attention ===\n";
    
    const int batch_size = 1;
    const int seq_len = 512;
    const int hidden_size = 768;
    const int num_heads = 12;
    const int head_dim = hidden_size / num_heads;
    const int num_runs = 50;
    
    // Create input tensors
    auto query = create_random_tensor({batch_size, num_heads, seq_len, head_dim});
    auto key = create_random_tensor({batch_size, num_heads, seq_len, head_dim});
    auto value = create_random_tensor({batch_size, num_heads, seq_len, head_dim});
    
    // Test Local Attention
    {
        std::cout << "Testing Local Attention (window=128)...\n";
        auto local_attn = std::make_unique<operators::attention::LocalAttention>(
            operators::attention::LocalAttention::Config{128, true, -1.0f});
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto output = local_attn->forward(*query, *key, *value);
        }
        
        // Benchmark
        SimpleTimer timer;
        std::vector<double> latencies;
        
        for (int i = 0; i < num_runs; ++i) {
            timer.reset();
            auto output = local_attn->forward(*query, *key, *value);
            latencies.push_back(timer.elapsed_ms());
        }
        
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double min_latency = *std::min_element(latencies.begin(), latencies.end());
        double max_latency = *std::max_element(latencies.begin(), latencies.end());
        
        std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean_latency << " ms\n";
        std::cout << "  Min:  " << min_latency << " ms\n";
        std::cout << "  Max:  " << max_latency << " ms\n";
        std::cout << "  Throughput: " << 1000.0 / mean_latency << " ops/sec\n\n";
    }
    
    // Test Strided Attention
    {
        std::cout << "Testing Strided Attention (stride=4)...\n";
        auto strided_attn = std::make_unique<operators::attention::StridedAttention>(
            operators::attention::StridedAttention::Config{4, 32, -1.0f});
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto output = strided_attn->forward(*query, *key, *value);
        }
        
        // Benchmark
        SimpleTimer timer;
        std::vector<double> latencies;
        
        for (int i = 0; i < num_runs; ++i) {
            timer.reset();
            auto output = strided_attn->forward(*query, *key, *value);
            latencies.push_back(timer.elapsed_ms());
        }
        
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double min_latency = *std::min_element(latencies.begin(), latencies.end());
        double max_latency = *std::max_element(latencies.begin(), latencies.end());
        
        std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean_latency << " ms\n";
        std::cout << "  Min:  " << min_latency << " ms\n";
        std::cout << "  Max:  " << max_latency << " ms\n";
        std::cout << "  Throughput: " << 1000.0 / mean_latency << " ops/sec\n\n";
    }
}

void benchmark_linear_attention() {
    std::cout << "=== Benchmarking Linear Attention ===\n";
    
    const int batch_size = 1;
    const int seq_len = 512;
    const int hidden_size = 768;
    const int num_heads = 12;
    const int head_dim = hidden_size / num_heads;
    const int num_runs = 50;
    
    // Create input tensors
    auto query = create_random_tensor({batch_size, num_heads, seq_len, head_dim});
    auto key = create_random_tensor({batch_size, num_heads, seq_len, head_dim});
    auto value = create_random_tensor({batch_size, num_heads, seq_len, head_dim});
    
    // Test Performer Attention
    {
        std::cout << "Testing Performer Attention (256 features)...\n";
        auto performer_attn = std::make_unique<operators::attention::PerformerAttention>(
            operators::attention::LinearAttentionBase::Config{operators::attention::LinearAttentionType::PERFORMER, 256, 256, -1.0f, false});
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto output = performer_attn->forward(*query, *key, *value);
        }
        
        // Benchmark
        SimpleTimer timer;
        std::vector<double> latencies;
        
        for (int i = 0; i < num_runs; ++i) {
            timer.reset();
            auto output = performer_attn->forward(*query, *key, *value);
            latencies.push_back(timer.elapsed_ms());
        }
        
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double min_latency = *std::min_element(latencies.begin(), latencies.end());
        double max_latency = *std::max_element(latencies.begin(), latencies.end());
        
        std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean_latency << " ms\n";
        std::cout << "  Min:  " << min_latency << " ms\n";
        std::cout << "  Max:  " << max_latency << " ms\n";
        std::cout << "  Throughput: " << 1000.0 / mean_latency << " ops/sec\n\n";
    }
    
    // Test Linformer Attention
    {
        std::cout << "Testing Linformer Attention (256 proj dim)...\n";
        auto linformer_attn = std::make_unique<operators::attention::LinformerAttention>(
            operators::attention::LinearAttentionBase::Config{operators::attention::LinearAttentionType::LINFORMER, 256, 256, -1.0f, false});
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto output = linformer_attn->forward(*query, *key, *value);
        }
        
        // Benchmark
        SimpleTimer timer;
        std::vector<double> latencies;
        
        for (int i = 0; i < num_runs; ++i) {
            timer.reset();
            auto output = linformer_attn->forward(*query, *key, *value);
            latencies.push_back(timer.elapsed_ms());
        }
        
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double min_latency = *std::min_element(latencies.begin(), latencies.end());
        double max_latency = *std::max_element(latencies.begin(), latencies.end());
        
        std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean_latency << " ms\n";
        std::cout << "  Min:  " << min_latency << " ms\n";
        std::cout << "  Max:  " << max_latency << " ms\n";
        std::cout << "  Throughput: " << 1000.0 / mean_latency << " ops/sec\n\n";
    }
}

void benchmark_multi_query_attention() {
    std::cout << "=== Benchmarking Multi-Query Attention ===\n";
    
    const int batch_size = 1;
    const int seq_len = 512;
    const int hidden_size = 768;
    const int num_heads = 12;
    const int head_dim = hidden_size / num_heads;
    const int num_runs = 50;
    
    // Test different KV head configurations
    std::vector<int> kv_heads_configs = {1, 2, 4, 8};
    
    for (int kv_heads : kv_heads_configs) {
        std::cout << "Testing Multi-Query Attention (KV heads=" << kv_heads << ")...\n";
        
        // Create input tensors
        auto query = create_random_tensor({batch_size, num_heads, seq_len, head_dim});
        auto key = create_random_tensor({batch_size, kv_heads, seq_len, head_dim});
        auto value = create_random_tensor({batch_size, kv_heads, seq_len, head_dim});
        auto mask = create_random_tensor({batch_size, 1, seq_len, seq_len});
        
        auto mqa_attn = std::make_unique<operators::attention::MultiQueryAttention>(
            operators::attention::MultiQueryAttention::Config{num_heads, head_dim, 64, true, false, false, -1.0f, 0.0f});
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto output = mqa_attn->forward(*query, *key, *value, *mask);
        }
        
        // Benchmark
        SimpleTimer timer;
        std::vector<double> latencies;
        
        for (int i = 0; i < num_runs; ++i) {
            timer.reset();
            auto output = mqa_attn->forward(*query, *key, *value, *mask);
            latencies.push_back(timer.elapsed_ms());
        }
        
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double min_latency = *std::min_element(latencies.begin(), latencies.end());
        double max_latency = *std::max_element(latencies.begin(), latencies.end());
        
        std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean_latency << " ms\n";
        std::cout << "  Min:  " << min_latency << " ms\n";
        std::cout << "  Max:  " << max_latency << " ms\n";
        std::cout << "  Throughput: " << 1000.0 / mean_latency << " ops/sec\n\n";
    }
}

void benchmark_ssm_models() {
    std::cout << "=== Benchmarking State Space Models ===\n";
    
    const int batch_size = 1;
    const int seq_len = 512;
    const int d_model = 768;
    const int d_state = 64;
    const int num_runs = 50;
    
    // Create input tensor
    auto input = create_random_tensor({batch_size, seq_len, d_model});
    
    // Test Mamba SSM
    {
        std::cout << "Testing Mamba SSM...\n";
        auto mamba_ssm = std::make_unique<operators::models::MambaSSM>(
            operators::models::StateSpaceModelBase::Config{operators::models::SSMType::MAMBA, d_model, d_state, 4, 2, false, false, 0.001f, 0.1f});
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto output = mamba_ssm->forward(*input);
        }
        
        // Benchmark
        SimpleTimer timer;
        std::vector<double> latencies;
        
        for (int i = 0; i < num_runs; ++i) {
            timer.reset();
            auto output = mamba_ssm->forward(*input);
            latencies.push_back(timer.elapsed_ms());
        }
        
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double min_latency = *std::min_element(latencies.begin(), latencies.end());
        double max_latency = *std::max_element(latencies.begin(), latencies.end());
        
        std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean_latency << " ms\n";
        std::cout << "  Min:  " << min_latency << " ms\n";
        std::cout << "  Max:  " << max_latency << " ms\n";
        std::cout << "  Throughput: " << 1000.0 / mean_latency << " ops/sec\n\n";
    }
    
    // Test S4 SSM
    {
        std::cout << "Testing S4 SSM...\n";
        operators::models::S4SSM::S4Config s4_config;
        s4_config.type = operators::models::SSMType::S4;
        s4_config.d_model = d_model;
        s4_config.d_state = d_state;
        s4_config.d_conv = 4;
        s4_config.expand_factor = 2;
        s4_config.structure = operators::models::S4SSM::StructureType::DPLR;
        s4_config.rank = 64;
        s4_config.use_fft = true;
        s4_config.trainable_dt = true;
        
        auto s4_ssm = std::make_unique<operators::models::S4SSM>(s4_config);
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto output = s4_ssm->forward(*input);
        }
        
        // Benchmark
        SimpleTimer timer;
        std::vector<double> latencies;
        
        for (int i = 0; i < num_runs; ++i) {
            timer.reset();
            auto output = s4_ssm->forward(*input);
            latencies.push_back(timer.elapsed_ms());
        }
        
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double min_latency = *std::min_element(latencies.begin(), latencies.end());
        double max_latency = *std::max_element(latencies.begin(), latencies.end());
        
        std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean_latency << " ms\n";
        std::cout << "  Min:  " << min_latency << " ms\n";
        std::cout << "  Max:  " << max_latency << " ms\n";
        std::cout << "  Throughput: " << 1000.0 / mean_latency << " ops/sec\n\n";
    }
    
    // Test S5 SSM
    {
        std::cout << "Testing S5 SSM...\n";
        auto s5_ssm = std::make_unique<operators::models::S5SSM>(
            operators::models::StateSpaceModelBase::Config{operators::models::SSMType::S5, d_model, d_state, 4, 2, false, false, 0.001f, 0.1f});
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto output = s5_ssm->forward(*input);
        }
        
        // Benchmark
        SimpleTimer timer;
        std::vector<double> latencies;
        
        for (int i = 0; i < num_runs; ++i) {
            timer.reset();
            auto output = s5_ssm->forward(*input);
            latencies.push_back(timer.elapsed_ms());
        }
        
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double min_latency = *std::min_element(latencies.begin(), latencies.end());
        double max_latency = *std::max_element(latencies.begin(), latencies.end());
        
        std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean_latency << " ms\n";
        std::cout << "  Min:  " << min_latency << " ms\n";
        std::cout << "  Max:  " << max_latency << " ms\n";
        std::cout << "  Throughput: " << 1000.0 / mean_latency << " ops/sec\n\n";
    }
}

void benchmark_mixture_of_experts() {
    std::cout << "=== Benchmarking Mixture of Experts ===\n";
    
    const int batch_size = 1;
    const int seq_len = 512;
    const int hidden_size = 768;
    const int num_runs = 50;
    
    // Create input tensor
    auto input = create_random_tensor({batch_size, seq_len, hidden_size});
    
    // Test different expert configurations
    std::vector<std::pair<int, int>> expert_configs = {{8, 2}, {16, 4}, {32, 8}};
    
    for (auto [num_experts, top_k] : expert_configs) {
        std::cout << "Testing MoE (experts=" << num_experts << ", top_k=" << top_k << ")...\n";
        
        auto moe = std::make_unique<operators::models::MixtureOfExperts>(
            operators::models::MixtureOfExperts::Config{operators::models::ExpertType::FEEDFORWARD, num_experts, hidden_size, hidden_size * 4, top_k, false, false, 0.1f});
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            auto output = moe->forward(*input);
        }
        
        // Benchmark
        SimpleTimer timer;
        std::vector<double> latencies;
        
        for (int i = 0; i < num_runs; ++i) {
            timer.reset();
            auto output = moe->forward(*input);
            latencies.push_back(timer.elapsed_ms());
        }
        
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double min_latency = *std::min_element(latencies.begin(), latencies.end());
        double max_latency = *std::max_element(latencies.begin(), latencies.end());
        
        std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean_latency << " ms\n";
        std::cout << "  Min:  " << min_latency << " ms\n";
        std::cout << "  Max:  " << max_latency << " ms\n";
        std::cout << "  Throughput: " << 1000.0 / mean_latency << " ops/sec\n\n";
    }
}

void print_banner() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DeepCpp Framework Benchmark                          ║
║                     Simple Component Performance Test                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;
}

void print_summary() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════════════════════╗
║                            Benchmark Complete                               ║
║                                                                              ║
║  All core components have been successfully benchmarked!                    ║
║  Results show the performance characteristics of each component.             ║
║                                                                              ║
║  For detailed analysis, consider running with different:                    ║
║  - Sequence lengths (128, 512, 1024, 2048)                                 ║
║  - Batch sizes (1, 4, 8, 16)                                               ║
║  - Model dimensions (512, 768, 1024, 1536)                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;
}

int main() {
    print_banner();
    
    try {
        benchmark_sparse_attention();
        benchmark_linear_attention();
        benchmark_multi_query_attention();
        benchmark_ssm_models();
        benchmark_mixture_of_experts();
        
        print_summary();
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 