#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <fstream>

// Include our benchmark and integration systems
#include "src/benchmarks/comprehensive_benchmark.h"
#include "src/integration/onnx_integration.h"

// ONNX Runtime for comparison
#include <onnxruntime_cxx_api.h>

using namespace deepcpp;

class ComprehensiveBenchmarkApp {
private:
    benchmarks::BenchmarkConfig config_;
    std::unique_ptr<benchmarks::ComprehensiveBenchmark> benchmark_;
    std::unique_ptr<integration::ONNXIntegrationManager> integration_manager_;
    
public:
    ComprehensiveBenchmarkApp() {
        // Default configuration
        config_.batch_size = 1;
        config_.sequence_length = 512;
        config_.hidden_size = 768;
        config_.num_heads = 12;
        config_.head_dim = 64;
        config_.num_warmup_runs = 10;
        config_.num_benchmark_runs = 100;
        config_.use_simd = true;
        config_.use_openmp = true;
        
        benchmark_ = std::make_unique<benchmarks::ComprehensiveBenchmark>(config_);
        integration_manager_ = std::make_unique<integration::ONNXIntegrationManager>();
    }
    
    void print_banner() {
        std::cout << R"(
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DeepCpp Framework Comprehensive Benchmark                 ║
║                                                                              ║
║  Testing: Sparse Attention, Linear Attention, Multi-Query Attention,        ║
║           State Space Models, Mixture of Experts, SIMD Optimizations        ║
║                                                                              ║
║  Integration: ONNX Runtime Custom Operators                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;
    }
    
    void print_config() {
        std::cout << "=== Benchmark Configuration ===\n";
        std::cout << "Batch Size:       " << config_.batch_size << "\n";
        std::cout << "Sequence Length:  " << config_.sequence_length << "\n";
        std::cout << "Hidden Size:      " << config_.hidden_size << "\n";
        std::cout << "Number of Heads:  " << config_.num_heads << "\n";
        std::cout << "Head Dimension:   " << config_.head_dim << "\n";
        std::cout << "Warmup Runs:      " << config_.num_warmup_runs << "\n";
        std::cout << "Benchmark Runs:   " << config_.num_benchmark_runs << "\n";
        std::cout << "SIMD Enabled:     " << (config_.use_simd ? "Yes" : "No") << "\n";
        std::cout << "OpenMP Enabled:   " << (config_.use_openmp ? "Yes" : "No") << "\n";
        std::cout << "================================\n\n";
    }
    
    void run_component_benchmarks() {
        std::cout << "Starting Component Benchmarks...\n\n";
        
        // Run all our custom component benchmarks
        benchmark_->run_all_benchmarks();
        
        // Save results
        benchmark_->save_results_csv("component_benchmark_results.csv");
        
        std::cout << "\nComponent benchmarks completed!\n";
        std::cout << "Results saved to: component_benchmark_results.csv\n\n";
    }
    
    void run_onnx_integration_benchmarks() {
        std::cout << "Starting ONNX Runtime Integration Benchmarks...\n\n";
        
        // Register our custom operators
        integration_manager_->register_all_operators();
        
        // Test with existing ONNX models if available
        std::vector<std::string> model_paths = {
            "models/simple_model.onnx",
            "models/transformer.onnx",
            "models/simple_transformer.onnx"
        };
        
        for (const auto& model_path : model_paths) {
            std::ifstream file(model_path);
            if (file.good()) {
                std::cout << "Testing ONNX model: " << model_path << "\n";
                benchmark_onnx_model(model_path);
            } else {
                std::cout << "Model not found: " << model_path << " (skipping)\n";
            }
        }
        
        std::cout << "\nONNX integration benchmarks completed!\n\n";
    }
    
    void benchmark_onnx_model(const std::string& model_path) {
        try {
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DeepCppBenchmark");
            Ort::SessionOptions session_options;
            
            // Configure for performance
            session_options.SetIntraOpNumThreads(8);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // Try to register our custom ops
            try {
                integration_manager_->RegisterCustomOps(&session_options, OrtGetApiBase());
                std::cout << "  Custom operators registered successfully\n";
            } catch (const std::exception& e) {
                std::cout << "  Warning: Could not register custom ops: " << e.what() << "\n";
            }
            
            // Create session
            Ort::Session session(env, model_path.c_str(), session_options);
            
            // Get input info
            size_t num_inputs = session.GetInputCount();
            std::cout << "  Number of inputs: " << num_inputs << "\n";
            
            // Create dummy inputs based on typical transformer shapes
            std::vector<Ort::Value> input_tensors;
            std::vector<const char*> input_names;
            
            for (size_t i = 0; i < num_inputs; ++i) {
                auto input_name = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions{});
                input_names.push_back(input_name.get());
                
                // Create dummy input tensor (assuming token IDs)
                std::vector<int64_t> input_shape = {config_.batch_size, config_.sequence_length};
                std::vector<int64_t> input_data(config_.batch_size * config_.sequence_length, 1);
                
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                    memory_info, input_data.data(), input_data.size(),
                    input_shape.data(), input_shape.size()));
            }
            
            // Get output names
            std::vector<const char*> output_names;
            size_t num_outputs = session.GetOutputCount();
            for (size_t i = 0; i < num_outputs; ++i) {
                auto output_name = session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions{});
                output_names.push_back(output_name.get());
            }
            
            // Warmup runs
            std::cout << "  Performing warmup runs...\n";
            for (int i = 0; i < config_.num_warmup_runs; ++i) {
                auto outputs = session.Run(Ort::RunOptions{nullptr},
                                         input_names.data(), input_tensors.data(), input_tensors.size(),
                                         output_names.data(), output_names.size());
            }
            
            // Benchmark runs
            std::cout << "  Performing benchmark runs...\n";
            std::vector<double> latencies;
            
            for (int i = 0; i < config_.num_benchmark_runs; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                
                auto outputs = session.Run(Ort::RunOptions{nullptr},
                                         input_names.data(), input_tensors.data(), input_tensors.size(),
                                         output_names.data(), output_names.size());
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                latencies.push_back(duration.count() / 1000.0);
            }
            
            // Calculate statistics
            double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
            double min_latency = *std::min_element(latencies.begin(), latencies.end());
            double max_latency = *std::max_element(latencies.begin(), latencies.end());
            
            std::cout << "  Results:\n";
            std::cout << "    Mean Latency: " << std::fixed << std::setprecision(3) << mean_latency << " ms\n";
            std::cout << "    Min Latency:  " << min_latency << " ms\n";
            std::cout << "    Max Latency:  " << max_latency << " ms\n";
            std::cout << "    Throughput:   " << 1000.0 / mean_latency << " inferences/sec\n\n";
            
        } catch (const std::exception& e) {
            std::cout << "  Error benchmarking model: " << e.what() << "\n\n";
        }
    }
    
    void run_scaling_analysis() {
        std::cout << "Starting Scaling Analysis...\n\n";
        
        // Test different configurations
        std::vector<benchmarks::BenchmarkConfig> configs = {
            {1, 128, 512, 8, 64, 50000, 8, 2, 5, 50, true, true},   // Small
            {1, 256, 768, 12, 64, 50000, 8, 2, 5, 50, true, true},  // Medium
            {1, 512, 768, 12, 64, 50000, 8, 2, 5, 50, true, true},  // Large
            {1, 1024, 1024, 16, 64, 50000, 16, 4, 5, 50, true, true}, // XL
        };
        
        std::vector<std::string> config_names = {"Small", "Medium", "Large", "XL"};
        
        for (size_t i = 0; i < configs.size(); ++i) {
            std::cout << "=== Testing " << config_names[i] << " Configuration ===\n";
            
            auto scaling_benchmark = std::make_unique<benchmarks::ComprehensiveBenchmark>(configs[i]);
            
            // Run a subset of benchmarks for scaling analysis
            scaling_benchmark->benchmark_sparse_attention();
            scaling_benchmark->benchmark_linear_attention();
            
            auto results = scaling_benchmark->get_results();
            
            std::cout << "Results for " << config_names[i] << ":\n";
            for (const auto& result : results) {
                std::cout << "  " << result.component_name << " - " << result.variant_name 
                          << ": " << result.mean_latency_ms << " ms\n";
            }
            std::cout << "\n";
            
            // Save individual results
            scaling_benchmark->save_results_csv("scaling_" + config_names[i] + "_results.csv");
        }
        
        std::cout << "Scaling analysis completed!\n\n";
    }
    
    void generate_performance_report() {
        std::cout << "Generating Performance Report...\n\n";
        
        auto results = benchmark_->get_results();
        
        // Generate comprehensive report
        std::ofstream report("performance_report.md");
        report << "# DeepCpp Framework Performance Report\n\n";
        report << "## Configuration\n";
        report << "- Batch Size: " << config_.batch_size << "\n";
        report << "- Sequence Length: " << config_.sequence_length << "\n";
        report << "- Hidden Size: " << config_.hidden_size << "\n";
        report << "- Number of Heads: " << config_.num_heads << "\n\n";
        
        report << "## Component Performance Summary\n\n";
        
        // Group results by component
        std::map<std::string, std::vector<benchmarks::BenchmarkResult>> grouped_results;
        for (const auto& result : results) {
            grouped_results[result.component_name].push_back(result);
        }
        
        for (const auto& [component, component_results] : grouped_results) {
            report << "### " << component << "\n\n";
            report << "| Variant | Mean Latency (ms) | Throughput (ops/sec) | Memory (MB) | GFLOPS |\n";
            report << "|---------|-------------------|---------------------|-------------|--------|\n";
            
            for (const auto& result : component_results) {
                report << "| " << result.variant_name 
                       << " | " << std::fixed << std::setprecision(3) << result.mean_latency_ms
                       << " | " << std::fixed << std::setprecision(1) << result.throughput_ops_per_sec
                       << " | " << std::fixed << std::setprecision(1) << result.memory_usage_mb
                       << " | " << std::fixed << std::setprecision(2) << result.flops_per_second / 1e9
                       << " |\n";
            }
            report << "\n";
        }
        
        report << "## Recommendations\n\n";
        
        // Find best performers
        for (const auto& [component, component_results] : grouped_results) {
            auto best = std::min_element(component_results.begin(), component_results.end(),
                [](const benchmarks::BenchmarkResult& a, const benchmarks::BenchmarkResult& b) {
                    return a.mean_latency_ms < b.mean_latency_ms;
                });
            
            report << "- **" << component << "**: Best performer is **" << best->variant_name 
                   << "** with " << best->mean_latency_ms << " ms latency\n";
        }
        
        report.close();
        
        std::cout << "Performance report generated: performance_report.md\n\n";
    }
    
    void run_all_benchmarks() {
        print_banner();
        print_config();
        
        // Run component benchmarks
        run_component_benchmarks();
        
        // Run ONNX integration benchmarks
        run_onnx_integration_benchmarks();
        
        // Run scaling analysis
        run_scaling_analysis();
        
        // Generate comprehensive report
        generate_performance_report();
        
        std::cout << "🎉 All benchmarks completed successfully!\n";
        std::cout << "\nGenerated files:\n";
        std::cout << "- component_benchmark_results.csv\n";
        std::cout << "- scaling_*_results.csv\n";
        std::cout << "- performance_report.md\n\n";
    }
    
    void set_config(const benchmarks::BenchmarkConfig& config) {
        config_ = config;
        benchmark_->set_config(config);
    }
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --batch-size <N>     Batch size (default: 1)\n";
    std::cout << "  --seq-len <N>        Sequence length (default: 512)\n";
    std::cout << "  --hidden-size <N>    Hidden size (default: 768)\n";
    std::cout << "  --num-heads <N>      Number of attention heads (default: 12)\n";
    std::cout << "  --warmup-runs <N>    Number of warmup runs (default: 10)\n";
    std::cout << "  --benchmark-runs <N> Number of benchmark runs (default: 100)\n";
    std::cout << "  --no-simd            Disable SIMD optimizations\n";
    std::cout << "  --no-openmp          Disable OpenMP parallelization\n";
    std::cout << "  --help               Show this help\n";
}

int main(int argc, char** argv) {
    ComprehensiveBenchmarkApp app;
    benchmarks::BenchmarkConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--batch-size" && i + 1 < argc) {
            config.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--seq-len" && i + 1 < argc) {
            config.sequence_length = std::stoi(argv[++i]);
        } else if (arg == "--hidden-size" && i + 1 < argc) {
            config.hidden_size = std::stoi(argv[++i]);
        } else if (arg == "--num-heads" && i + 1 < argc) {
            config.num_heads = std::stoi(argv[++i]);
        } else if (arg == "--warmup-runs" && i + 1 < argc) {
            config.num_warmup_runs = std::stoi(argv[++i]);
        } else if (arg == "--benchmark-runs" && i + 1 < argc) {
            config.num_benchmark_runs = std::stoi(argv[++i]);
        } else if (arg == "--no-simd") {
            config.use_simd = false;
        } else if (arg == "--no-openmp") {
            config.use_openmp = false;
        } else {
            std::cout << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Update head dimension based on hidden size and num heads
    config.head_dim = config.hidden_size / config.num_heads;
    
    app.set_config(config);
    app.run_all_benchmarks();
    
    return 0;
} 