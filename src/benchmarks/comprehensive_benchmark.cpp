#include "comprehensive_benchmark.h"
#include <random>
#include <cmath>
#include <sstream>
#include <fstream>

namespace deepcpp {
namespace benchmarks {

std::shared_ptr<core::Tensor> ComprehensiveBenchmark::create_random_tensor(
    const std::vector<int>& shape, core::DataType dtype) {
    
    auto tensor = std::make_shared<core::Tensor>(shape, dtype);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    float* data = tensor->data_ptr<float>();
    size_t total_size = tensor->numel();
    
    for (size_t i = 0; i < total_size; ++i) {
        data[i] = dis(gen);
    }
    
    return tensor;
}

double ComprehensiveBenchmark::estimate_memory_usage(
    const std::vector<std::shared_ptr<core::Tensor>>& tensors) {
    
    size_t total_bytes = 0;
    for (const auto& tensor : tensors) {
        total_bytes += tensor->numel() * sizeof(float); // Assuming float32
    }
    return total_bytes / (1024.0 * 1024.0); // Convert to MB
}

double ComprehensiveBenchmark::calculate_attention_flops(
    int batch_size, int seq_len, int hidden_size, int num_heads) {
    
    int head_dim = hidden_size / num_heads;
    // QK^T: B * H * S * S * D
    // Softmax: B * H * S * S (approx)
    // AV: B * H * S * S * D
    // Total: 2 * B * H * S * S * D + B * H * S * S
    return 2.0 * batch_size * num_heads * seq_len * seq_len * head_dim + 
           batch_size * num_heads * seq_len * seq_len;
}

double ComprehensiveBenchmark::calculate_linear_flops(
    int batch_size, int seq_len, int input_size, int output_size) {
    return 2.0 * batch_size * seq_len * input_size * output_size;
}

double ComprehensiveBenchmark::calculate_ssm_flops(
    int batch_size, int seq_len, int state_size) {
    // Simplified SSM FLOPS estimation
    return batch_size * seq_len * state_size * 10; // Rough estimate
}

double ComprehensiveBenchmark::calculate_moe_flops(
    int batch_size, int seq_len, int hidden_size, int num_experts, int top_k) {
    // Router + Expert computation
    double router_flops = calculate_linear_flops(batch_size, seq_len, hidden_size, num_experts);
    double expert_flops = calculate_linear_flops(batch_size, seq_len, hidden_size, hidden_size * 4) * top_k;
    return router_flops + expert_flops;
}

void ComprehensiveBenchmark::benchmark_sparse_attention() {
    std::cout << "=== Benchmarking Sparse Attention Variants ===\n";
    
    // Test different sparse attention patterns
    std::vector<std::string> patterns = {
        "local", "strided", "bigbird", "longformer", "block_sparse", "adaptive"
    };
    
    for (const auto& pattern : patterns) {
        std::cout << "Testing " << pattern << " attention...\n";
        
        // Create input tensors
        auto query = create_random_tensor({config_.batch_size, config_.num_heads, 
                                         config_.sequence_length, config_.head_dim});
        auto key = create_random_tensor({config_.batch_size, config_.num_heads, 
                                       config_.sequence_length, config_.head_dim});
        auto value = create_random_tensor({config_.batch_size, config_.num_heads, 
                                         config_.sequence_length, config_.head_dim});
        
        // Create appropriate sparse attention instance
        std::unique_ptr<operators::attention::SparseAttentionBase> sparse_attn;
        
        if (pattern == "local") {
            sparse_attn = std::make_unique<operators::attention::LocalAttention>(
                config_.hidden_size, config_.num_heads, 128); // window_size = 128
        } else if (pattern == "strided") {
            sparse_attn = std::make_unique<operators::attention::StridedAttention>(
                config_.hidden_size, config_.num_heads, 4); // stride = 4
        } else if (pattern == "bigbird") {
            sparse_attn = std::make_unique<operators::attention::BigBirdAttention>(
                config_.hidden_size, config_.num_heads, 64, 3, 2); // block_size, num_random, num_global
        }
        // Add other patterns...
        
        if (!sparse_attn) continue;
        
        // Warmup runs
        for (int i = 0; i < config_.num_warmup_runs; ++i) {
            auto output = sparse_attn->forward(*query, *key, *value);
        }
        
        // Benchmark runs
        std::vector<double> latencies;
        Timer timer;
        
        for (int i = 0; i < config_.num_benchmark_runs; ++i) {
            timer.reset();
            auto output = sparse_attn->forward(*query, *key, *value);
            latencies.push_back(timer.elapsed_ms());
        }
        
        // Calculate statistics
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double variance = 0.0;
        for (double lat : latencies) {
            variance += (lat - mean_latency) * (lat - mean_latency);
        }
        double std_latency = std::sqrt(variance / latencies.size());
        double min_latency = *std::min_element(latencies.begin(), latencies.end());
        double max_latency = *std::max_element(latencies.begin(), latencies.end());
        
        // Create result
        BenchmarkResult result;
        result.component_name = "SparseAttention";
        result.variant_name = pattern;
        result.mean_latency_ms = mean_latency;
        result.std_latency_ms = std_latency;
        result.min_latency_ms = min_latency;
        result.max_latency_ms = max_latency;
        result.throughput_ops_per_sec = 1000.0 / mean_latency;
        result.memory_usage_mb = estimate_memory_usage({query, key, value});
        result.flops_per_second = calculate_attention_flops(
            config_.batch_size, config_.sequence_length, 
            config_.hidden_size, config_.num_heads) / (mean_latency / 1000.0);
        result.config = config_;
        
        results_.push_back(result);
        result.print();
    }
}

void ComprehensiveBenchmark::benchmark_linear_attention() {
    std::cout << "=== Benchmarking Linear Attention Variants ===\n";
    
    std::vector<std::string> variants = {
        "performer", "linformer", "cosformer", "synthesizer", "random_features"
    };
    
    for (const auto& variant : variants) {
        std::cout << "Testing " << variant << " attention...\n";
        
        auto query = create_random_tensor({config_.batch_size, config_.num_heads, 
                                         config_.sequence_length, config_.head_dim});
        auto key = create_random_tensor({config_.batch_size, config_.num_heads, 
                                       config_.sequence_length, config_.head_dim});
        auto value = create_random_tensor({config_.batch_size, config_.num_heads, 
                                         config_.sequence_length, config_.head_dim});
        
        std::unique_ptr<operators::attention::LinearAttentionBase> linear_attn;
        
        if (variant == "performer") {
            linear_attn = std::make_unique<operators::attention::PerformerAttention>(
                config_.hidden_size, config_.num_heads, 256); // num_features
        } else if (variant == "linformer") {
            linear_attn = std::make_unique<operators::attention::LinformerAttention>(
                config_.hidden_size, config_.num_heads, 256); // projected_dim
        }
        // Add other variants...
        
        if (!linear_attn) continue;
        
        // Benchmark similar to sparse attention
        std::vector<double> latencies;
        
        // Warmup
        for (int i = 0; i < config_.num_warmup_runs; ++i) {
            auto output = linear_attn->forward(*query, *key, *value);
        }
        
        // Benchmark
        Timer timer;
        for (int i = 0; i < config_.num_benchmark_runs; ++i) {
            timer.reset();
            auto output = linear_attn->forward(*query, *key, *value);
            latencies.push_back(timer.elapsed_ms());
        }
        
        // Calculate and store results
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        
        BenchmarkResult result;
        result.component_name = "LinearAttention";
        result.variant_name = variant;
        result.mean_latency_ms = mean_latency;
        result.throughput_ops_per_sec = 1000.0 / mean_latency;
        result.memory_usage_mb = estimate_memory_usage({query, key, value});
        result.flops_per_second = calculate_attention_flops(
            config_.batch_size, config_.sequence_length, 
            config_.hidden_size, config_.num_heads) / (mean_latency / 1000.0);
        result.config = config_;
        
        results_.push_back(result);
        result.print();
    }
}

void ComprehensiveBenchmark::benchmark_multi_query_attention() {
    std::cout << "=== Benchmarking Multi-Query Attention Variants ===\n";
    
    std::vector<int> kv_head_configs = {1, 2, 4, 8}; // Different KV head counts
    
    for (int num_kv_heads : kv_head_configs) {
        std::cout << "Testing MQA with " << num_kv_heads << " KV heads...\n";
        
        auto query = create_random_tensor({config_.batch_size, config_.num_heads, 
                                         config_.sequence_length, config_.head_dim});
        auto key = create_random_tensor({config_.batch_size, num_kv_heads, 
                                       config_.sequence_length, config_.head_dim});
        auto value = create_random_tensor({config_.batch_size, num_kv_heads, 
                                         config_.sequence_length, config_.head_dim});
        
        auto mqa = std::make_unique<operators::attention::MultiQueryAttention>(
            config_.hidden_size, config_.num_heads, config_.num_kv_heads);
        
        std::vector<double> latencies;
        
        // Warmup
        for (int i = 0; i < config_.num_warmup_runs; ++i) {
            auto output = mqa->forward(*query, *key, *value);
        }
        
        // Benchmark
        Timer timer;
        for (int i = 0; i < config_.num_benchmark_runs; ++i) {
            timer.reset();
            auto output = mqa->forward(*query, *key, *value);
            latencies.push_back(timer.elapsed_ms());
        }
        
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        
        BenchmarkResult result;
        result.component_name = "MultiQueryAttention";
        result.variant_name = "KV_heads_" + std::to_string(num_kv_heads);
        result.mean_latency_ms = mean_latency;
        result.throughput_ops_per_sec = 1000.0 / mean_latency;
        result.memory_usage_mb = estimate_memory_usage({query, key, value});
        result.config = config_;
        
        results_.push_back(result);
        result.print();
    }
}

void ComprehensiveBenchmark::benchmark_ssm_models() {
    std::cout << "=== Benchmarking State Space Models ===\n";
    
    std::vector<std::string> ssm_variants = {"mamba", "s4", "s5", "gated", "bidirectional"};
    
    for (const auto& variant : ssm_variants) {
        std::cout << "Testing " << variant << " SSM...\n";
        
        auto input = create_random_tensor({config_.batch_size, config_.sequence_length, config_.hidden_size});
        
        std::unique_ptr<operators::models::StateSpaceModelBase> ssm;
        
        if (variant == "mamba") {
            ssm = std::make_unique<operators::models::MambaSSM>(config_.hidden_size, 64); // state_size
        } else if (variant == "s4") {
            ssm = std::make_unique<operators::models::S4SSM>(config_.hidden_size, 64);
        } else if (variant == "s5") {
            ssm = std::make_unique<operators::models::S5SSM>(config_.hidden_size, 64);
        }
        // Add other variants...
        
        if (!ssm) continue;
        
        std::vector<double> latencies;
        
        // Warmup
        for (int i = 0; i < config_.num_warmup_runs; ++i) {
            auto output = ssm->forward(*input);
        }
        
        // Benchmark
        Timer timer;
        for (int i = 0; i < config_.num_benchmark_runs; ++i) {
            timer.reset();
            auto output = ssm->forward(*input);
            latencies.push_back(timer.elapsed_ms());
        }
        
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        
        BenchmarkResult result;
        result.component_name = "StateSpaceModel";
        result.variant_name = variant;
        result.mean_latency_ms = mean_latency;
        result.throughput_ops_per_sec = 1000.0 / mean_latency;
        result.memory_usage_mb = estimate_memory_usage({input});
        result.flops_per_second = calculate_ssm_flops(
            config_.batch_size, config_.sequence_length, 64) / (mean_latency / 1000.0);
        result.config = config_;
        
        results_.push_back(result);
        result.print();
    }
}

void ComprehensiveBenchmark::benchmark_mixture_of_experts() {
    std::cout << "=== Benchmarking Mixture of Experts ===\n";
    
    std::vector<int> expert_counts = {4, 8, 16, 32};
    std::vector<int> top_k_values = {1, 2, 4};
    
    for (int num_experts : expert_counts) {
        for (int top_k : top_k_values) {
            if (top_k > num_experts) continue;
            
            std::cout << "Testing MoE with " << num_experts << " experts, top-" << top_k << "...\n";
            
            auto input = create_random_tensor({config_.batch_size, config_.sequence_length, config_.hidden_size});
            
            auto moe = std::make_unique<operators::models::MixtureOfExperts>(
                config_.hidden_size, config_.hidden_size * 4, num_experts, top_k);
            
            std::vector<double> latencies;
            
            // Warmup
            for (int i = 0; i < config_.num_warmup_runs; ++i) {
                auto output = moe->forward(*input);
            }
            
            // Benchmark
            Timer timer;
            for (int i = 0; i < config_.num_benchmark_runs; ++i) {
                timer.reset();
                auto output = moe->forward(*input);
                latencies.push_back(timer.elapsed_ms());
            }
            
            double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
            
            BenchmarkResult result;
            result.component_name = "MixtureOfExperts";
            result.variant_name = "E" + std::to_string(num_experts) + "_K" + std::to_string(top_k);
            result.mean_latency_ms = mean_latency;
            result.throughput_ops_per_sec = 1000.0 / mean_latency;
            result.memory_usage_mb = estimate_memory_usage({input});
            result.flops_per_second = calculate_moe_flops(
                config_.batch_size, config_.sequence_length, 
                config_.hidden_size, num_experts, top_k) / (mean_latency / 1000.0);
            result.config = config_;
            
            results_.push_back(result);
            result.print();
        }
    }
}

void ComprehensiveBenchmark::run_all_benchmarks() {
    std::cout << "Starting Comprehensive Deep Learning Framework Benchmark\n";
    std::cout << "========================================================\n\n";
    
    results_.clear();
    
    benchmark_sparse_attention();
    benchmark_linear_attention();
    benchmark_multi_query_attention();
    benchmark_ssm_models();
    benchmark_mixture_of_experts();
    
    std::cout << "\n=== Benchmark Complete ===\n";
    print_summary();
}

void ComprehensiveBenchmark::save_results_csv(const std::string& filename) {
    std::ofstream file(filename);
    
    // Header
    file << "Component,Variant,MeanLatency(ms),StdLatency(ms),MinLatency(ms),MaxLatency(ms),"
         << "Throughput(ops/sec),Memory(MB),FLOPS,BatchSize,SeqLength,HiddenSize,NumHeads\n";
    
    // Data
    for (const auto& result : results_) {
        file << result.component_name << "," << result.variant_name << ","
             << result.mean_latency_ms << "," << result.std_latency_ms << ","
             << result.min_latency_ms << "," << result.max_latency_ms << ","
             << result.throughput_ops_per_sec << "," << result.memory_usage_mb << ","
             << result.flops_per_second << "," << result.config.batch_size << ","
             << result.config.sequence_length << "," << result.config.hidden_size << ","
             << result.config.num_heads << "\n";
    }
    
    std::cout << "Results saved to " << filename << "\n";
}

void ComprehensiveBenchmark::print_summary() {
    std::cout << "\n=== BENCHMARK SUMMARY ===\n";
    std::cout << "Total components tested: " << results_.size() << "\n\n";
    
    // Group by component
    std::map<std::string, std::vector<BenchmarkResult>> grouped;
    for (const auto& result : results_) {
        grouped[result.component_name].push_back(result);
    }
    
    for (const auto& [component, results] : grouped) {
        std::cout << component << ":\n";
        
        // Find best performer
        auto best = std::min_element(results.begin(), results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.mean_latency_ms < b.mean_latency_ms;
            });
        
        std::cout << "  Best: " << best->variant_name 
                  << " (" << best->mean_latency_ms << " ms)\n";
        
        // Calculate average
        double avg_latency = 0.0;
        for (const auto& r : results) {
            avg_latency += r.mean_latency_ms;
        }
        avg_latency /= results.size();
        
        std::cout << "  Average: " << avg_latency << " ms\n\n";
    }
}

} // namespace benchmarks
} // namespace deepcpp 