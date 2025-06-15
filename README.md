# C++DL: Native C++ Deep Learning Framework

A high-performance, educational deep learning framework built from scratch in C++ with modern architecture implementations and cross-platform compatibility.

## 🎯 Purpose & Vision

This framework was created to address several critical challenges in the deep learning ecosystem:

### 🖥️ **macOS Intel Compatibility First**
- **No More Platform Headaches**: Built specifically to run seamlessly on macOS Intel without hitting the typical bugs, kernel issues, or compatibility problems that plague other frameworks
- **Native Performance**: Leverages C++ for optimal performance on Intel architectures without relying on problematic CUDA/ROCm installations
- **Zero External Dependencies**: No PyTorch, TensorFlow, or other heavyweight frameworks that often break on macOS Intel systems

### 📚 **Educational Deep Learning Architecture**
- **Learn by Building**: Understand modern architectures (Transformers, Mamba, MoE) by implementing them from scratch rather than using black-box libraries
- **Mathematical Transparency**: Every operation is implemented explicitly - see exactly how attention mechanisms, state space models, and mixture of experts work under the hood
- **No Magic**: Unlike frameworks that hide complexity, every component is readable, modifiable, and educational

### ⚡ **Performance Without Compromise**
- **C++ Speed**: Native C++ performance with SIMD optimizations where beneficial
- **Python Integration Ready**: Designed to interface with Python for data preprocessing and visualization while keeping compute in C++
- **Memory Efficient**: Direct memory management without garbage collection overhead
- **Deployment Ready**: Compile to native binaries for production deployment

## 🏗️ **Architecture Overview**

### Core Components
- **Linear Attention (FAVOR+)**: Efficient attention mechanism with kernel approximation
- **Mamba State Space Models**: Selective state space models for sequence modeling
- **Mixture of Experts (MoE)**: Sparse expert routing for scalable model capacity
- **Custom Training Pipeline**: Full gradient computation and optimization from scratch

### Modern Features
- **Tokenization**: Custom vocabulary building and text processing
- **Data Generation**: Synthetic data creation for rapid prototyping
- **Training Loop**: Complete training pipeline with evaluation metrics
- **Model Persistence**: Save/load trained models for inference

## 🚀 **Getting Started**

### Prerequisites
- macOS (Intel optimized, but works on Apple Silicon too)
- CMake 3.15+
- C++17 compatible compiler (Clang recommended)

### Quick Build
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run Custom Training Demo
```bash
./custom_training_demo
```

## 💡 **Use Cases**

### 1. **Research & Experimentation**
- Implement new architectures without framework limitations
- Modify core components for research purposes
- Understand exactly how modern ML architectures work

### 2. **Production Deployment**
- Deploy models without Python runtime dependencies
- Native performance for inference servers
- Minimal memory footprint for edge deployment

### 3. **Educational Purposes**
- Learn deep learning by implementing from scratch
- Understand the mathematics behind modern architectures
- Bridge the gap between theory and implementation

### 4. **macOS Intel Development**
- Avoid compatibility issues with CUDA-dependent frameworks
- Native development environment without virtualization
- Consistent performance across Intel Mac hardware

## 🔧 **Technical Highlights**

### Hand-Built Architectures
- **FAVOR+ Linear Attention**: Kernel approximation for O(n) attention complexity
- **Mamba SSM**: Selective scan mechanism with state-dependent transitions
- **Mixture of Experts**: Top-k routing with load balancing
- **Adam Optimizer**: Full gradient-based optimization with momentum

### Performance Optimizations
- **SIMD-Ready**: Vectorized operations where beneficial
- **Memory Efficient**: Careful memory management and reuse
- **Batch Processing**: Efficient batched operations
- **Gradient Computation**: Automatic differentiation for all components

### Real Implementation Details
- **No Placeholders**: Every component is fully implemented
- **Mathematical Accuracy**: Proper implementation of research papers
- **Numerical Stability**: Careful handling of floating-point operations
- **Modular Design**: Easy to extend and modify components

## 📊 **Example: Sentiment Analysis Training**

The included demo trains a sentiment classifier using:
- **Model**: 2.69M parameters with Embedding → Linear Attention → Mamba → MoE → Classifier
- **Data**: Synthetic sentiment data with positive/negative word patterns
- **Training**: 3 epochs with Adam optimizer and learning rate scheduling
- **Evaluation**: Real-time accuracy and F1 score tracking

```cpp
// Simple usage example
SentimentModel model(config);
CustomTrainer trainer(model, train_data, val_data);
trainer.train(3); // Train for 3 epochs
```

## 🎓 **Learning Benefits**

### Understanding Modern Architectures
- **Attention Mechanisms**: See how FAVOR+ approximates full attention
- **State Space Models**: Understand selective scan and state transitions
- **Mixture of Experts**: Learn expert routing and load balancing
- **Optimization**: Implement gradient descent and momentum from scratch

### C++ Deep Learning Skills
- **Memory Management**: Learn efficient tensor operations
- **Performance Optimization**: Understand SIMD and vectorization
- **System Design**: Build scalable ML systems in C++
- **Cross-Platform Development**: Write portable high-performance code

## 🔮 **Future Roadmap**

- [ ] GPU acceleration with Metal Performance Shaders (macOS native)
- [ ] Python bindings for easy data preprocessing
- [ ] More architecture implementations (Vision Transformers, etc.)
- [ ] Distributed training support
- [ ] Model quantization and optimization
- [ ] WebAssembly compilation for browser deployment

## 🤝 **Contributing**

This framework is designed to be educational and extensible. Contributions welcome for:
- New architecture implementations
- Performance optimizations
- Educational examples and tutorials
- macOS-specific optimizations
- Documentation improvements

## 📝 **License**

MIT License - Feel free to use for research, education, or commercial purposes.

---

**Built with ❤️ for the macOS Intel community and deep learning enthusiasts who want to understand how things really work under the hood.** 