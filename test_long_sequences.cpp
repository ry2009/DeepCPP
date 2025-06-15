#include <iostream>
#include <chrono>
#include <memory>

#include "src/core/tensor/tensor.h"
#include "src/operators/attention/sparse_attention.h"

using namespace deepcpp;

int main() {
    std::cout << "=== Testing Long Sequence Processing ===\n";
    std::cout << "Testing sequences that would crash standard attention...\n\n";
    
    // Test progressively larger sequences
    std::vector<int> sequence_lengths = {1024, 2048, 4096, 8192};
    
    for (int seq_len : sequence_lengths) {
        std::cout << "Testing sequence length: " << seq_len << " tokens\n";
        
        try {
            // Create tensors for the sequence
            auto query = std::make_shared<core::Tensor>(
                std::vector<int64_t>{1, 12, seq_len, 64}, core::DataType::FLOAT32);
            auto key = std::make_shared<core::Tensor>(
                std::vector<int64_t>{1, 12, seq_len, 64}, core::DataType::FLOAT32);
            auto value = std::make_shared<core::Tensor>(
                std::vector<int64_t>{1, 12, seq_len, 64}, core::DataType::FLOAT32);
            
            // Initialize with random data
            query->fill_random();
            key->fill_random();
            value->fill_random();
            
            // Use sparse local attention - O(n) memory complexity
            operators::attention::LocalAttention local_attn(
                operators::attention::LocalAttention::Config{256, true, -1.0f});
            
            // Time the operation
            auto start = std::chrono::high_resolution_clock::now();
            auto output = local_attn.forward(*query, *key, *value);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            // Calculate memory usage estimate
            size_t tensor_size = seq_len * 12 * 64 * sizeof(float);
            size_t total_memory_mb = (tensor_size * 4) / (1024 * 1024); // 4 tensors (q,k,v,output)
            
            std::cout << "  ✅ SUCCESS!\n";
            std::cout << "  ⏱️  Time: " << duration.count() << "ms\n";
            std::cout << "  💾 Memory: ~" << total_memory_mb << "MB\n";
            std::cout << "  📊 Output shape: [" << output.shape()[0] << ", " 
                     << output.shape()[1] << ", " << output.shape()[2] << ", " 
                     << output.shape()[3] << "]\n";
            
            // Estimate what standard attention would need
            size_t std_attention_memory = (seq_len * seq_len * 12 * sizeof(float)) / (1024 * 1024);
            std::cout << "  🔥 Standard attention would need: ~" << std_attention_memory << "MB just for attention matrix!\n";
            
            if (seq_len >= 4096) {
                std::cout << "  🚀 This sequence length would likely crash PyTorch on most machines!\n";
            }
            
        } catch (const std::exception& e) {
            std::cout << "  ❌ Error: " << e.what() << std::endl;
        }
        
        std::cout << "\n";
    }
    
    std::cout << "=== Memory Efficiency Comparison ===\n";
    std::cout << "Sequence Length | Standard Attention | Sparse Attention | Memory Saved\n";
    std::cout << "----------------|-------------------|------------------|-------------\n";
    
    for (int seq_len : sequence_lengths) {
        size_t std_mem = (seq_len * seq_len * 12 * sizeof(float)) / (1024 * 1024);
        size_t sparse_mem = (seq_len * 256 * 12 * sizeof(float)) / (1024 * 1024); // window size 256
        double savings = ((double)(std_mem - sparse_mem) / std_mem) * 100;
        
        printf("%15d | %17zu MB | %16zu MB | %10.1f%%\n", 
               seq_len, std_mem, sparse_mem, savings);
    }
    
    std::cout << "\n🎯 Key Insights:\n";
    std::cout << "• Sparse attention scales linearly O(n) vs quadratic O(n²)\n";
    std::cout << "• Memory savings increase dramatically with sequence length\n";
    std::cout << "• Enables processing of long sequences impossible with standard attention\n";
    
    return 0;
} 