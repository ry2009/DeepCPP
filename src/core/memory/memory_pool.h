#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace deepcpp {

/**
 * High-performance memory pool for efficient allocation/deallocation
 * 
 * Features:
 * - Thread-safe allocation and deallocation
 * - Multiple pool sizes for different allocation patterns
 * - Memory alignment support for SIMD operations
 * - Memory usage tracking and statistics
 * - Automatic pool expansion when needed
 * - Memory defragmentation capabilities
 */
class MemoryPool {
public:
    struct Config {
        size_t initial_size;
        size_t max_size;
        size_t alignment;
        size_t block_size;
        bool allow_expansion;
        bool track_statistics;
        
        Config() : initial_size(1024 * 1024), max_size(1024 * 1024 * 1024),
                   alignment(32), block_size(64), allow_expansion(true),
                   track_statistics(true) {}
    };
    
    /**
     * Constructor
     * @param config Memory pool configuration
     */
    explicit MemoryPool(const Config& config = Config());
    
    /**
     * Constructor with simple size specification
     * @param size Initial pool size in bytes
     */
    explicit MemoryPool(size_t size);
    
    ~MemoryPool();
    
    /**
     * Allocate aligned memory block
     * @param size Size in bytes to allocate
     * @param alignment Memory alignment (0 = use default)
     * @return Pointer to allocated memory, nullptr on failure
     */
    void* allocate(size_t size, size_t alignment = 0);
    
    /**
     * Deallocate memory block
     * @param ptr Pointer to memory block to deallocate
     */
    void deallocate(void* ptr);
    
    /**
     * Reallocate memory block with new size
     * @param ptr Existing memory block
     * @param new_size New size in bytes
     * @return Pointer to reallocated memory
     */
    void* reallocate(void* ptr, size_t new_size);
    
    /**
     * Clear all allocations and reset pool
     */
    void clear();
    
    /**
     * Defragment memory pool to reduce fragmentation
     */
    void defragment();
    
    /**
     * Get current memory usage statistics
     */
         struct Statistics {
         size_t total_allocated;
         size_t total_deallocated;
         size_t current_usage;
         size_t peak_usage;
         size_t pool_size;
         size_t num_allocations;
         size_t num_deallocations;
         size_t fragmentation_ratio;
         double allocation_efficiency;
         
         Statistics() : total_allocated(0), total_deallocated(0), current_usage(0),
                       peak_usage(0), pool_size(0), num_allocations(0),
                       num_deallocations(0), fragmentation_ratio(0),
                       allocation_efficiency(0.0) {}
     };
    
    Statistics getStatistics() const;
    void resetStatistics();
    
    /**
     * Check if pointer was allocated from this pool
     */
    bool owns(void* ptr) const;
    
    /**
     * Get size of allocated block
     */
    size_t getAllocatedSize(void* ptr) const;
    
    /**
     * Set memory allocation callbacks for debugging
     */
    using AllocCallback = std::function<void(void*, size_t)>;
    using DeallocCallback = std::function<void(void*, size_t)>;
    
    void setCallbacks(AllocCallback alloc_cb, DeallocCallback dealloc_cb);

private:
    struct Block {
        void* ptr;
        size_t size;
        bool is_free;
        Block* next;
        Block* prev;
        
        Block(void* p, size_t s) : ptr(p), size(s), is_free(true), next(nullptr), prev(nullptr) {}
    };
    
    struct Pool {
        void* memory;
        size_t size;
        size_t used;
        Block* free_list;
        Block* used_list;
        
        Pool(size_t s);
        ~Pool();
    };
    
    Config config_;
    std::vector<std::unique_ptr<Pool>> pools_;
    mutable std::mutex mutex_;
    
    // Statistics tracking
    mutable Statistics stats_;
    std::unordered_map<void*, size_t> allocation_sizes_;
    
    // Callbacks
    AllocCallback alloc_callback_;
    DeallocCallback dealloc_callback_;
    
    // Private methods
    void* allocateFromPool(Pool* pool, size_t size, size_t alignment);
    Block* findFreeBlock(Pool* pool, size_t size, size_t alignment);
    Block* splitBlock(Block* block, size_t size);
    void mergeBlocks(Pool* pool);
    void expandPool(size_t min_size);
    size_t alignSize(size_t size, size_t alignment) const;
    void updateStatistics(size_t allocated, bool is_allocation);
    
    // Memory alignment utilities
    static void* alignPointer(void* ptr, size_t alignment);
    static size_t getAlignmentOffset(void* ptr, size_t alignment);
};

/**
 * Thread-local memory pool for high-performance single-threaded access
 */
class ThreadLocalMemoryPool {
public:
    explicit ThreadLocalMemoryPool(size_t pool_size = 1024 * 1024);
    ~ThreadLocalMemoryPool();
    
    void* allocate(size_t size, size_t alignment = 32);
    void deallocate(void* ptr);
    void clear();
    
    static ThreadLocalMemoryPool& getInstance();

private:
    std::unique_ptr<MemoryPool> pool_;
    static thread_local std::unique_ptr<ThreadLocalMemoryPool> instance_;
};

/**
 * RAII memory allocator for automatic cleanup
 */
template<typename T>
class PoolAllocator {
public:
    using value_type = T;
    
    explicit PoolAllocator(MemoryPool* pool) : pool_(pool) {}
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U>& other) : pool_(other.pool_) {}
    
    T* allocate(size_t n) {
        return static_cast<T*>(pool_->allocate(n * sizeof(T), alignof(T)));
    }
    
    void deallocate(T* ptr, size_t n) {
        (void)n; // Suppress unused parameter warning
        pool_->deallocate(ptr);
    }
    
    template<typename U>
    bool operator==(const PoolAllocator<U>& other) const {
        return pool_ == other.pool_;
    }
    
    template<typename U>
    bool operator!=(const PoolAllocator<U>& other) const {
        return !(*this == other);
    }

private:
    MemoryPool* pool_;
    
    template<typename U>
    friend class PoolAllocator;
};

/**
 * Smart pointer with pool-based allocation
 */
template<typename T>
class PoolPtr {
public:
    PoolPtr() : ptr_(nullptr), pool_(nullptr) {}
    
    PoolPtr(T* ptr, MemoryPool* pool) : ptr_(ptr), pool_(pool) {}
    
    ~PoolPtr() {
        if (ptr_ && pool_) {
            ptr_->~T();
            pool_->deallocate(ptr_);
        }
    }
    
    // Move semantics
    PoolPtr(PoolPtr&& other) noexcept : ptr_(other.ptr_), pool_(other.pool_) {
        other.ptr_ = nullptr;
        other.pool_ = nullptr;
    }
    
    PoolPtr& operator=(PoolPtr&& other) noexcept {
        if (this != &other) {
            if (ptr_ && pool_) {
                ptr_->~T();
                pool_->deallocate(ptr_);
            }
            ptr_ = other.ptr_;
            pool_ = other.pool_;
            other.ptr_ = nullptr;
            other.pool_ = nullptr;
        }
        return *this;
    }
    
    // Disable copy semantics
    PoolPtr(const PoolPtr&) = delete;
    PoolPtr& operator=(const PoolPtr&) = delete;
    
    T* get() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    
    explicit operator bool() const { return ptr_ != nullptr; }
    
    void reset() {
        if (ptr_ && pool_) {
            ptr_->~T();
            pool_->deallocate(ptr_);
        }
        ptr_ = nullptr;
        pool_ = nullptr;
    }

private:
    T* ptr_;
    MemoryPool* pool_;
};

/**
 * Factory function for creating pool-allocated objects
 */
template<typename T, typename... Args>
PoolPtr<T> makePoolPtr(MemoryPool* pool, Args&&... args) {
    void* memory = pool->allocate(sizeof(T), alignof(T));
    if (!memory) {
        return PoolPtr<T>();
    }
    
    T* ptr = new(memory) T(std::forward<Args>(args)...);
    return PoolPtr<T>(ptr, pool);
}

/**
 * Global memory pool manager for framework-wide memory management
 */
class GlobalMemoryManager {
public:
    static GlobalMemoryManager& getInstance();
    
    /**
     * Get memory pool for specific usage pattern
     */
    MemoryPool* getTensorPool();      // For tensor allocations
    MemoryPool* getWorkspacePool();   // For temporary workspace
    MemoryPool* getKVCachePool();     // For KV cache storage
    MemoryPool* getGeneralPool();     // For general allocations
    
    /**
     * Configure memory pools
     */
    void configurePools(const MemoryPool::Config& tensor_config,
                       const MemoryPool::Config& workspace_config,
                       const MemoryPool::Config& cache_config);
    
    /**
     * Get global memory statistics
     */
         struct GlobalStats {
         MemoryPool::Statistics tensor_stats;
         MemoryPool::Statistics workspace_stats;
         MemoryPool::Statistics cache_stats;
         MemoryPool::Statistics general_stats;
         size_t total_memory_usage;
         size_t peak_memory_usage;
         
         GlobalStats() : total_memory_usage(0), peak_memory_usage(0) {}
     };
    
    GlobalStats getGlobalStatistics() const;
    void resetAllStatistics();
    
    /**
     * Memory cleanup and optimization
     */
    void cleanup();
    void defragmentAll();

private:
    GlobalMemoryManager();
    ~GlobalMemoryManager();
    
    std::unique_ptr<MemoryPool> tensor_pool_;
    std::unique_ptr<MemoryPool> workspace_pool_;
    std::unique_ptr<MemoryPool> kv_cache_pool_;
    std::unique_ptr<MemoryPool> general_pool_;
    
    mutable std::mutex mutex_;
    mutable size_t peak_memory_usage_;
};

} // namespace deepcpp 