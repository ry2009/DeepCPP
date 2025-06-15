#pragma once

#include <chrono>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

// Include all our new components
#include "../operators/attention/sparse_attention.h"
#include "../operators/attention/linear_attention.h"
#include "../operators/attention/multi_query_attention.h"
#include "../operators/models/ssm.h"
#include "../operators/models/mixture_of_experts.h"
#include "../operators/performance/simd_kernels.h"
#include "../core/tensor/tensor.h"

namespace deepcpp {
namespace benchmarks {

struct BenchmarkConfig {
    int batch_size = 1;
    int sequence_length = 512;
    int hidden_size = 768;
    int num_heads = 12;
    int head_dim = 64;
    int vocab_size = 50000;
    int num_experts = 8;
    int top_k = 2;
    int num_warmup_runs = 10;
    int num_benchmark_runs = 100;
    bool use_simd = true;
    bool use_openmp = true;
};

struct BenchmarkResult {
    std::string component_name;
    std::string variant_name;
    double mean_latency_ms;
    double std_latency_ms;
    double min_latency_ms;
    double max_latency_ms;
    double throughput_ops_per_sec;
    double memory_usage_mb;
    double flops_per_second;
    BenchmarkConfig config;
    
    void print() const {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "=== " << component_name << " - " << variant_name << " ===\n";
        std::cout << "Mean Latency: " << mean_latency_ms << " ms\n";
        std::cout << "Std Latency:  " << std_latency_ms << " ms\n";
        std::cout << "Min Latency:  " << min_latency_ms << " ms\n";
        std::cout << "Max Latency:  " << max_latency_ms << " ms\n";
        std::cout << "Throughput:   " << throughput_ops_per_sec << " ops/sec\n";
        std::cout << "Memory:       " << memory_usage_mb << " MB\n";
        std::cout << "FLOPS:        " << flops_per_second / 1e9 << " GFLOPS\n";
        std::cout << "Config: B=" << config.batch_size << ", S=" << config.sequence_length 
                  << ", H=" << config.hidden_size << ", NH=" << config.num_heads << "\n\n";
    }
};

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0;
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

class ComprehensiveBenchmark {
private:
    BenchmarkConfig config_;
    std::vector<BenchmarkResult> results_;
    
    // Helper to create random tensors
    std::shared_ptr<core::Tensor> create_random_tensor(const std::vector<int>& shape, 
                                                      core::DataType dtype = core::DataType::FLOAT32);
    
    // Memory usage estimation
    double estimate_memory_usage(const std::vector<std::shared_ptr<core::Tensor>>& tensors);
    
    // FLOPS calculation helpers
    double calculate_attention_flops(int batch_size, int seq_len, int hidden_size, int num_heads);
    double calculate_linear_flops(int batch_size, int seq_len, int input_size, int output_size);
    double calculate_ssm_flops(int batch_size, int seq_len, int state_size);
    double calculate_moe_flops(int batch_size, int seq_len, int hidden_size, int num_experts, int top_k);
    
public:
    ComprehensiveBenchmark(const BenchmarkConfig& config = BenchmarkConfig{}) 
        : config_(config) {}
    
    // Attention benchmarks
    void benchmark_sparse_attention();
    void benchmark_linear_attention();
    void benchmark_multi_query_attention();
    
    // Model architecture benchmarks
    void benchmark_ssm_models();
    void benchmark_mixture_of_experts();
    
    // Performance optimization benchmarks
    void benchmark_simd_kernels();
    void benchmark_memory_optimizations();
    
    // Comparison benchmarks
    void benchmark_attention_comparison();
    void benchmark_scaling_analysis();
    
    // Run all benchmarks
    void run_all_benchmarks();
    
    // Results management
    const std::vector<BenchmarkResult>& get_results() const { return results_; }
    void save_results_csv(const std::string& filename);
    void save_results_json(const std::string& filename);
    void print_summary();
    void print_detailed_results();
    
    // Configuration
    void set_config(const BenchmarkConfig& config) { config_ = config; }
    const BenchmarkConfig& get_config() const { return config_; }
};

// Specialized benchmark classes for different components
class AttentionBenchmark {
private:
    BenchmarkConfig config_;
    
public:
    AttentionBenchmark(const BenchmarkConfig& config) : config_(config) {}
    
    BenchmarkResult benchmark_flash_attention();
    BenchmarkResult benchmark_sparse_attention(const std::string& pattern_type);
    BenchmarkResult benchmark_linear_attention(const std::string& kernel_type);
    BenchmarkResult benchmark_multi_query_attention(int num_kv_heads);
    
    std::vector<BenchmarkResult> compare_all_attention_variants();
};

class ModelBenchmark {
private:
    BenchmarkConfig config_;
    
public:
    ModelBenchmark(const BenchmarkConfig& config) : config_(config) {}
    
    BenchmarkResult benchmark_mamba_ssm();
    BenchmarkResult benchmark_s4_ssm();
    BenchmarkResult benchmark_mixture_of_experts();
    BenchmarkResult benchmark_transformer_block();
    
    std::vector<BenchmarkResult> compare_all_model_variants();
};

class PerformanceBenchmark {
private:
    BenchmarkConfig config_;
    
public:
    PerformanceBenchmark(const BenchmarkConfig& config) : config_(config) {}
    
    BenchmarkResult benchmark_simd_matmul();
    BenchmarkResult benchmark_simd_attention();
    BenchmarkResult benchmark_memory_bandwidth();
    BenchmarkResult benchmark_cache_efficiency();
    
    std::vector<BenchmarkResult> compare_optimization_levels();
};

// Utility functions for benchmark analysis
namespace analysis {
    double calculate_speedup(const BenchmarkResult& baseline, const BenchmarkResult& optimized);
    double calculate_efficiency(const BenchmarkResult& result, double theoretical_peak_flops);
    void generate_performance_report(const std::vector<BenchmarkResult>& results, 
                                   const std::string& output_file);
    void plot_scaling_curves(const std::vector<BenchmarkResult>& results);
}

} // namespace benchmarks
} // namespace deepcpp 