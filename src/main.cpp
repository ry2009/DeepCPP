#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

// Function to load custom ops (will be linked from custom_ops.dylib)
extern "C" {
    OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
}

class InferenceSession {
public:
    InferenceSession(const std::string& model_path, int num_threads = 8) 
        : env_(ORT_LOGGING_LEVEL_WARNING, "DeepCppInference") {
        
        // Configure session options
        session_options_.SetIntraOpNumThreads(num_threads);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Enable oneDNN (DNNL) execution provider for CPU optimization
        try {
            // Try to enable DNNL EP - API varies by version
            std::vector<std::string> providers = Ort::GetAvailableProviders();
            bool dnnl_available = false;
            for (const auto& provider : providers) {
                if (provider == "DnnlExecutionProvider") {
                    dnnl_available = true;
                    break;
                }
            }
            
            if (dnnl_available) {
                session_options_.AppendExecutionProvider("DnnlExecutionProvider");
                std::cout << "oneDNN execution provider enabled\n";
            } else {
                std::cout << "oneDNN execution provider not available\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not enable oneDNN EP: " << e.what() << "\n";
        }
        
        // Add custom ops domain (optional - commented out for now)
        // TODO: Implement proper custom op registration for this ONNX Runtime version
        // try {
        //     RegisterCustomOps(session_options_, OrtGetApiBase());
        //     std::cout << "Custom ops registered successfully\n";
        // } catch (const std::exception& e) {
        //     std::cout << "Warning: Could not register custom ops: " << e.what() << "\n";
        // }
        
        // Create session
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        
        // Get input/output info
        input_names_ = getInputNames();
        output_names_ = getOutputNames();
        
        std::cout << "Model loaded: " << model_path << "\n";
        std::cout << "Inputs: ";
        for (const auto& name : input_names_) {
            std::cout << name << " ";
        }
        std::cout << "\nOutputs: ";
        for (const auto& name : output_names_) {
            std::cout << name << " ";
        }
        std::cout << "\n";
    }
    
    std::vector<Ort::Value> run(const std::vector<Ort::Value>& inputs) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Convert input names to const char*
        std::vector<const char*> input_names_cstr;
        for (const auto& name : input_names_) {
            input_names_cstr.push_back(name.c_str());
        }
        
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names_) {
            output_names_cstr.push_back(name.c_str());
        }
        
        // Run inference
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                   input_names_cstr.data(), inputs.data(), inputs.size(),
                                   output_names_cstr.data(), output_names_.size());
        
        auto end = std::chrono::high_resolution_clock::now();
        last_inference_time_ = std::chrono::duration<double, std::milli>(end - start).count();
        
        return outputs;
    }
    
    double getLastInferenceTime() const {
        return last_inference_time_;
    }
    
    const std::vector<std::string>& getInputNames() const {
        return input_names_;
    }
    
    const std::vector<std::string>& getOutputNames() const {
        return output_names_;
    }
    
    // Make session accessible for input type checking
    std::unique_ptr<Ort::Session> session_;

private:
    std::vector<std::string> getInputNames() {
        std::vector<std::string> names;
        size_t count = session_->GetInputCount();
        for (size_t i = 0; i < count; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator_);
            names.push_back(name.get());
        }
        return names;
    }
    
    std::vector<std::string> getOutputNames() {
        std::vector<std::string> names;
        size_t count = session_->GetOutputCount();
        for (size_t i = 0; i < count; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator_);
            names.push_back(name.get());
        }
        return names;
    }

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    double last_inference_time_;
};

// Helper function to create dummy input tensors based on model requirements
std::vector<Ort::Value> createDummyInputs(const InferenceSession& session, const std::vector<int64_t>& shape) {
    std::vector<Ort::Value> inputs;
    
    // Calculate total elements
    size_t total_elements = 1;
    for (auto dim : shape) {
        total_elements *= dim;
    }
    
    // Get input type info from the session
    auto input_type_info = session.session_->GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto element_type = tensor_info.GetElementType();
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        // Create token ID data (random token IDs within vocab range)
        std::vector<int64_t> data(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            data[i] = rand() % 1000;  // Random token ID [0, 999]
        }
        inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, data.data(), total_elements,
                                                             shape.data(), shape.size()));
    } else {
        // Create float data (random values between -1 and 1)
        std::vector<float> data(total_elements);
        for (size_t i = 0; i < total_elements; ++i) {
            data[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;  // Random float [-1, 1]
        }
        inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info, data.data(), total_elements,
                                                            shape.data(), shape.size()));
    }
    
    return inputs;
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_path> [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --batch-size <N>     Batch size (default: 1)\n";
    std::cout << "  --seq-len <N>        Sequence length (default: 512)\n"; 
    std::cout << "  --d-model <N>        Model dimension (default: 768)\n";
    std::cout << "  --num-runs <N>       Number of inference runs (default: 10)\n";
    std::cout << "  --threads <N>        Number of threads (default: 8)\n";
    std::cout << "  --help               Show this help\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    
    // Parse command line arguments
    int batch_size = 1;
    int seq_len = 512;
    int d_model = 768;
    int num_runs = 10;
    int num_threads = 8;
    
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        
        std::string arg = argv[i];
        if (arg == "--batch-size") {
            batch_size = std::stoi(argv[i + 1]);
        } else if (arg == "--seq-len") {
            seq_len = std::stoi(argv[i + 1]);
        } else if (arg == "--d-model") {
            d_model = std::stoi(argv[i + 1]);
        } else if (arg == "--num-runs") {
            num_runs = std::stoi(argv[i + 1]);
        } else if (arg == "--threads") {
            num_threads = std::stoi(argv[i + 1]);
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    try {
        // Check if model file exists
        std::ifstream file(model_path);
        if (!file.good()) {
            std::cerr << "Error: Model file not found: " << model_path << "\n";
            return 1;
        }
        
        std::cout << "=== Deep C++ Inference Framework ===\n";
        std::cout << "Model: " << model_path << "\n";
        std::cout << "Batch size: " << batch_size << "\n";
        std::cout << "Sequence length: " << seq_len << "\n";
        std::cout << "Model dimension: " << d_model << "\n";
        std::cout << "Threads: " << num_threads << "\n";
        std::cout << "Runs: " << num_runs << "\n\n";
        
        // Create inference session
        InferenceSession session(model_path, num_threads);
        
        // Create dummy input - adjust shape based on model type
        std::vector<int64_t> input_shape;
        if (model_path.find("transformer") != std::string::npos) {
            // Transformer models typically expect [batch_size, seq_len] for token IDs
            input_shape = {batch_size, seq_len};
        } else {
            // Other models might expect [batch_size, seq_len, d_model] for embeddings
            input_shape = {batch_size, seq_len, d_model};
        }
        auto inputs = createDummyInputs(session, input_shape);
        
        // Warmup run
        std::cout << "Warming up...\n";
        auto outputs = session.run(inputs);
        std::cout << "Warmup completed in " << session.getLastInferenceTime() << " ms\n\n";
        
        // Benchmark runs
        std::cout << "Running benchmark...\n";
        std::vector<double> times;
        
        for (int i = 0; i < num_runs; ++i) {
            outputs = session.run(inputs);
            double time = session.getLastInferenceTime();
            times.push_back(time);
            std::cout << "Run " << (i + 1) << "/" << num_runs << ": " << time << " ms\n";
        }
        
        // Calculate statistics
        double total_time = 0.0;
        double min_time = times[0];
        double max_time = times[0];
        
        for (double time : times) {
            total_time += time;
            min_time = std::min(min_time, time);  
            max_time = std::max(max_time, time);
        }
        
        double avg_time = total_time / num_runs;
        
        std::cout << "\n=== Results ===\n";
        std::cout << "Average: " << avg_time << " ms\n";
        std::cout << "Min: " << min_time << " ms\n";
        std::cout << "Max: " << max_time << " ms\n";
        std::cout << "Throughput: " << (1000.0 / avg_time) << " inferences/sec\n";
        std::cout << "Tokens/sec: " << (batch_size * seq_len * 1000.0 / avg_time) << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
} 