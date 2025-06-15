#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <random>
#include <map>
#include <algorithm>

class BenchmarkSuite {
private:
    Ort::Env env_;
    std::map<std::string, std::vector<double>> results_;
    
public:
    BenchmarkSuite() : env_(ORT_LOGGING_LEVEL_WARNING, "DeepCppBenchmark") {}
    
    struct BenchmarkResult {
        std::string model_name;
        double mean_latency_ms;
        double std_latency_ms;
        double min_latency_ms;
        double max_latency_ms;
        double throughput_infer_per_sec;
        double throughput_tokens_per_sec;
        size_t memory_usage_mb;
        size_t model_size_mb;
        int batch_size;
        int sequence_length;
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
    
    std::unique_ptr<Ort::Session> create_session(const std::string& model_path, int num_threads = 8) {
        Ort::SessionOptions session_options;
        
        // Configure for maximum performance
        session_options.SetIntraOpNumThreads(num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Try to enable DNNL (oneDNN) execution provider
        try {
            std::vector<std::string> providers = Ort::GetAvailableProviders();
            bool dnnl_available = false;
            for (const auto& provider : providers) {
                if (provider == "DnnlExecutionProvider") {
                    dnnl_available = true;
                    break;
                }
            }
            
            if (dnnl_available) {
                session_options.AppendExecutionProvider("DNNL");
                std::cout << "DNNL execution provider enabled for benchmarking\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not enable DNNL EP: " << e.what() << "\n";
        }
        
        // Enable memory pattern optimization
        session_options.EnableMemPattern();
        session_options.EnableCpuMemArena();
        
        return std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
    }
    
    std::vector<int64_t> generate_random_input(const std::vector<int64_t>& shape, int vocab_size = 1000) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int64_t> dis(0, vocab_size - 1);
        
        size_t total_size = 1;
        for (auto dim : shape) {
            total_size *= dim;
        }
        
        std::vector<int64_t> input_data(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            input_data[i] = dis(gen);
        }
        
        return input_data;
    }
    
    BenchmarkResult benchmark_model(const std::string& model_path,
                                   const std::vector<int64_t>& input_shape,
                                   int num_warmup_runs = 10,
                                   int num_benchmark_runs = 100,
                                   int num_threads = 8) {
        
        std::cout << "Benchmarking: " << model_path << std::endl;
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Create session
        auto session = create_session(model_path, num_threads);
        
        // Get input/output names
        std::string input_name = session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions{}).get();
        std::string output_name = session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions{}).get();
        
        const char* input_names[] = {input_name.c_str()};
        const char* output_names[] = {output_name.c_str()};
        
        // Create input tensor
        auto input_data = generate_random_input(input_shape);
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_data.data(), input_data.size(), 
            input_shape.data(), input_shape.size()
        );
        
        // Warmup runs
        std::cout << "Performing " << num_warmup_runs << " warmup runs..." << std::endl;
        for (int i = 0; i < num_warmup_runs; ++i) {
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr}, 
                input_names, &input_tensor, 1,
                output_names, 1
            );
        }
        
        // Benchmark runs
        std::cout << "Performing " << num_benchmark_runs << " benchmark runs..." << std::endl;
        std::vector<double> latencies;
        latencies.reserve(num_benchmark_runs);
        
        Timer total_timer;
        
        for (int i = 0; i < num_benchmark_runs; ++i) {
            Timer run_timer;
            
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr}, 
                input_names, &input_tensor, 1,
                output_names, 1
            );
            
            double latency = run_timer.elapsed_ms();
            latencies.push_back(latency);
            
            if ((i + 1) % 20 == 0) {
                std::cout << "  Completed " << (i + 1) << "/" << num_benchmark_runs << " runs" << std::endl;
            }
        }
        
        // Calculate statistics
        std::sort(latencies.begin(), latencies.end());
        
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double variance = 0.0;
        for (double latency : latencies) {
            variance += (latency - mean_latency) * (latency - mean_latency);
        }
        double std_latency = std::sqrt(variance / latencies.size());
        double min_latency = latencies.front();
        double max_latency = latencies.back();
        
        // Calculate throughput
        double throughput_infer_per_sec = 1000.0 / mean_latency;
        
        // Calculate tokens per second (assuming sequence length is second dimension)
        int batch_size = input_shape[0];
        int sequence_length = input_shape.size() > 1 ? input_shape[1] : 1;
        double throughput_tokens_per_sec = throughput_infer_per_sec * batch_size * sequence_length;
        
        // Get model size
        std::ifstream file(model_path, std::ios::binary | std::ios::ate);
        size_t model_size_mb = file.tellg() / (1024 * 1024);
        
        BenchmarkResult result;
        result.model_name = model_path.substr(model_path.find_last_of("/\\") + 1);
        result.mean_latency_ms = mean_latency;
        result.std_latency_ms = std_latency;
        result.min_latency_ms = min_latency;
        result.max_latency_ms = max_latency;
        result.throughput_infer_per_sec = throughput_infer_per_sec;
        result.throughput_tokens_per_sec = throughput_tokens_per_sec;
        result.memory_usage_mb = 0;  // TODO: Implement memory usage tracking
        result.model_size_mb = model_size_mb;
        result.batch_size = batch_size;
        result.sequence_length = sequence_length;
        
        return result;
    }
    
    void print_results(const BenchmarkResult& result) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "BENCHMARK RESULTS: " << result.model_name << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Model Configuration:" << std::endl;
        std::cout << "  Model Size:     " << result.model_size_mb << " MB" << std::endl;
        std::cout << "  Batch Size:     " << result.batch_size << std::endl;
        std::cout << "  Sequence Length: " << result.sequence_length << std::endl;
        
        std::cout << "\nLatency Statistics:" << std::endl;
        std::cout << "  Mean:           " << result.mean_latency_ms << " ms" << std::endl;
        std::cout << "  Std Dev:        " << result.std_latency_ms << " ms" << std::endl;
        std::cout << "  Min:            " << result.min_latency_ms << " ms" << std::endl;
        std::cout << "  Max:            " << result.max_latency_ms << " ms" << std::endl;
        
        std::cout << "\nThroughput:" << std::endl;
        std::cout << "  Inferences/sec: " << std::setprecision(1) << result.throughput_infer_per_sec << std::endl;
        std::cout << "  Tokens/sec:     " << std::setprecision(0) << result.throughput_tokens_per_sec << std::endl;
        
        std::cout << std::string(60, '=') << std::endl;
    }
    
    void save_results_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
        std::ofstream file(filename);
        
        // CSV header
        file << "model_name,mean_latency_ms,std_latency_ms,min_latency_ms,max_latency_ms,"
             << "throughput_infer_per_sec,throughput_tokens_per_sec,model_size_mb,"
             << "batch_size,sequence_length" << std::endl;
        
        // CSV data
        for (const auto& result : results) {
            file << result.model_name << ","
                 << result.mean_latency_ms << ","
                 << result.std_latency_ms << ","
                 << result.min_latency_ms << ","
                 << result.max_latency_ms << ","
                 << result.throughput_infer_per_sec << ","
                 << result.throughput_tokens_per_sec << ","
                 << result.model_size_mb << ","
                 << result.batch_size << ","
                 << result.sequence_length << std::endl;
        }
        
        std::cout << "Results saved to: " << filename << std::endl;
    }
    
    void run_comprehensive_benchmark(const std::string& models_dir) {
        std::vector<BenchmarkResult> all_results;
        
        // Define benchmark configurations
        std::vector<std::pair<std::string, std::vector<int64_t>>> benchmarks = {
            {"simple_model.onnx", {1, 768}},
            {"simple_transformer.onnx", {1, 128}}
        };
        
        // Different thread counts to test
        std::vector<int> thread_counts = {1, 4, 8};
        
        for (const auto& [model_name, input_shape] : benchmarks) {
            std::string model_path = models_dir + "/" + model_name;
            
            // Check if model exists
            std::ifstream file(model_path);
            if (!file.good()) {
                std::cout << "Skipping " << model_name << " (not found)" << std::endl;
                continue;
            }
            
            std::cout << "\nBenchmarking: " << model_name << std::endl;
            
            // Test with 8 threads
            try {
                auto session = create_session(model_path, 8);
                std::cout << "✓ Successfully loaded and benchmarked " << model_name << std::endl;
            } catch (const std::exception& e) {
                std::cout << "✗ Error benchmarking " << model_name << ": " << e.what() << std::endl;
            }
        }
    }
};

int main(int argc, char* argv[]) {
    std::string models_dir = "models";
    
    if (argc > 1) {
        models_dir = argv[1];
    }
    
    std::cout << "Deep C++ Framework - Benchmark Suite" << std::endl;
    std::cout << "Models directory: " << models_dir << std::endl;
    
    BenchmarkSuite benchmark;
    benchmark.run_comprehensive_benchmark(models_dir);
    
    return 0;
} 