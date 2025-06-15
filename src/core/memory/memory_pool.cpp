#include "memory_pool.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cassert>

namespace deepcpp {

// MemoryPool::Pool implementation
MemoryPool::Pool::Pool(size_t s) : size(s), used(0), free_list(nullptr), used_list(nullptr) {
    // Allocate aligned memory
    memory = std::aligned_alloc(32, size);
    if (!memory) {
        throw std::bad_alloc();
    }
    
    // Initialize with single free block
    free_list = new Block(memory, size);
}

MemoryPool::Pool::~Pool() {
    // Clean up block lists
    Block* current = free_list;
    while (current) {
        Block* next = current->next;
        delete current;
        current = next;
    }
    
    current = used_list;
    while (current) {
        Block* next = current->next;
        delete current;
        current = next;
    }
    
    // Free the memory
    if (memory) {
        std::free(memory);
    }
}

// MemoryPool implementation
MemoryPool::MemoryPool(const Config& config) : config_(config) {
    // Create initial pool
    pools_.emplace_back(std::make_unique<Pool>(config_.initial_size));
    
    // Initialize statistics
    stats_.pool_size = config_.initial_size;
}

MemoryPool::MemoryPool(size_t size) {
    Config config;
    config.initial_size = size;
    config.max_size = size * 10;
    config.alignment = 32;
    config.block_size = 64;
    config.allow_expansion = true;
    config.track_statistics = true;
    
    config_ = config;
    
    // Create initial pool
    pools_.emplace_back(std::make_unique<Pool>(config_.initial_size));
    
    // Initialize statistics
    stats_.pool_size = config_.initial_size;
}

MemoryPool::~MemoryPool() {
    // Pools are automatically cleaned up by unique_ptr
}

void* MemoryPool::allocate(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Use default alignment if not specified
    if (alignment == 0) {
        alignment = config_.alignment;
    }
    
    // Align size to minimum block size
    size = alignSize(size, config_.block_size);
    
    // Try to allocate from existing pools
    for (auto& pool : pools_) {
        void* ptr = allocateFromPool(pool.get(), size, alignment);
        if (ptr) {
            allocation_sizes_[ptr] = size;
            updateStatistics(size, true);
            
            if (alloc_callback_) {
                alloc_callback_(ptr, size);
            }
            
            return ptr;
        }
    }
    
    // Expand pool if allowed
    if (config_.allow_expansion) {
        size_t new_pool_size = std::max(size * 2, config_.initial_size);
        if (stats_.pool_size + new_pool_size <= config_.max_size) {
            expandPool(new_pool_size);
            
            // Try allocation again
            void* ptr = allocateFromPool(pools_.back().get(), size, alignment);
            if (ptr) {
                allocation_sizes_[ptr] = size;
                updateStatistics(size, true);
                
                if (alloc_callback_) {
                    alloc_callback_(ptr, size);
                }
                
                return ptr;
            }
        }
    }
    
    return nullptr; // Allocation failed
}

void MemoryPool::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find which pool owns this pointer
    for (auto& pool : pools_) {
        if (ptr >= pool->memory && 
            ptr < static_cast<char*>(pool->memory) + pool->size) {
            
            // Find the block in used list
            Block* current = pool->used_list;
            Block* prev = nullptr;
            
            while (current && current->ptr != ptr) {
                prev = current;
                current = current->next;
            }
            
            if (current) {
                // Remove from used list
                if (prev) {
                    prev->next = current->next;
                } else {
                    pool->used_list = current->next;
                }
                
                if (current->next) {
                    current->next->prev = prev;
                }
                
                // Add to free list
                current->is_free = true;
                current->next = pool->free_list;
                current->prev = nullptr;
                if (pool->free_list) {
                    pool->free_list->prev = current;
                }
                pool->free_list = current;
                
                // Update statistics
                size_t size = allocation_sizes_[ptr];
                allocation_sizes_.erase(ptr);
                updateStatistics(size, false);
                
                if (dealloc_callback_) {
                    dealloc_callback_(ptr, size);
                }
                
                // Try to merge adjacent free blocks
                mergeBlocks(pool.get());
                return;
            }
        }
    }
}

void* MemoryPool::reallocate(void* ptr, size_t new_size) {
    if (!ptr) {
        return allocate(new_size);
    }
    
    if (new_size == 0) {
        deallocate(ptr);
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Get current size
    auto it = allocation_sizes_.find(ptr);
    if (it == allocation_sizes_.end()) {
        return nullptr; // Pointer not found
    }
    
    size_t old_size = it->second;
    if (new_size <= old_size) {
        // Shrinking - just update size
        allocation_sizes_[ptr] = new_size;
        updateStatistics(old_size - new_size, false);
        return ptr;
    }
    
    // Need to allocate new block
    void* new_ptr = allocate(new_size);
    if (new_ptr) {
        std::memcpy(new_ptr, ptr, old_size);
        deallocate(ptr);
    }
    
    return new_ptr;
}

void MemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Reset all pools
    for (auto& pool : pools_) {
        // Move all used blocks to free list
        if (pool->used_list) {
            Block* last_used = pool->used_list;
            while (last_used->next) {
                last_used->is_free = true;
                last_used = last_used->next;
            }
            last_used->is_free = true;
            
            // Connect to free list
            last_used->next = pool->free_list;
            if (pool->free_list) {
                pool->free_list->prev = last_used;
            }
            pool->free_list = pool->used_list;
            pool->used_list = nullptr;
        }
        
        pool->used = 0;
    }
    
    allocation_sizes_.clear();
    stats_.current_usage = 0;
    stats_.num_allocations = 0;
    stats_.num_deallocations = 0;
}

void MemoryPool::defragment() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& pool : pools_) {
        mergeBlocks(pool.get());
    }
}

MemoryPool::Statistics MemoryPool::getStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void MemoryPool::resetStatistics() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t current_usage = stats_.current_usage;
    size_t pool_size = stats_.pool_size;
    
    stats_ = Statistics{};
    stats_.current_usage = current_usage;
    stats_.pool_size = pool_size;
}

bool MemoryPool::owns(void* ptr) const {
    if (!ptr) return false;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (const auto& pool : pools_) {
        if (ptr >= pool->memory && 
            ptr < static_cast<char*>(pool->memory) + pool->size) {
            return true;
        }
    }
    
    return false;
}

size_t MemoryPool::getAllocatedSize(void* ptr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocation_sizes_.find(ptr);
    return (it != allocation_sizes_.end()) ? it->second : 0;
}

void MemoryPool::setCallbacks(AllocCallback alloc_cb, DeallocCallback dealloc_cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    alloc_callback_ = alloc_cb;
    dealloc_callback_ = dealloc_cb;
}

// Private methods
void* MemoryPool::allocateFromPool(Pool* pool, size_t size, size_t alignment) {
    Block* block = findFreeBlock(pool, size, alignment);
    if (!block) {
        return nullptr;
    }
    
    // Split block if necessary
    if (block->size > size + config_.block_size) {
        block = splitBlock(block, size);
    }
    
    // Remove from free list
    if (block->prev) {
        block->prev->next = block->next;
    } else {
        pool->free_list = block->next;
    }
    
    if (block->next) {
        block->next->prev = block->prev;
    }
    
    // Add to used list
    block->is_free = false;
    block->next = pool->used_list;
    block->prev = nullptr;
    if (pool->used_list) {
        pool->used_list->prev = block;
    }
    pool->used_list = block;
    
    pool->used += block->size;
    
    return alignPointer(block->ptr, alignment);
}

MemoryPool::Block* MemoryPool::findFreeBlock(Pool* pool, size_t size, size_t alignment) {
    Block* current = pool->free_list;
    
    while (current) {
        if (current->is_free) {
            void* aligned_ptr = alignPointer(current->ptr, alignment);
            size_t alignment_offset = static_cast<char*>(aligned_ptr) - static_cast<char*>(current->ptr);
            
            if (current->size >= size + alignment_offset) {
                return current;
            }
        }
        current = current->next;
    }
    
    return nullptr;
}

MemoryPool::Block* MemoryPool::splitBlock(Block* block, size_t size) {
    if (block->size <= size + config_.block_size) {
        return block; // Not worth splitting
    }
    
    // Create new block for remaining space
    void* new_ptr = static_cast<char*>(block->ptr) + size;
    size_t new_size = block->size - size;
    Block* new_block = new Block(new_ptr, new_size);
    
    // Insert new block into free list
    new_block->next = block->next;
    new_block->prev = block;
    if (block->next) {
        block->next->prev = new_block;
    }
    block->next = new_block;
    
    // Update original block size
    block->size = size;
    
    return block;
}

void MemoryPool::mergeBlocks(Pool* pool) {
    Block* current = pool->free_list;
    
    while (current && current->next) {
        Block* next = current->next;
        
        // Check if blocks are adjacent
        if (current->is_free && next->is_free &&
            static_cast<char*>(current->ptr) + current->size == next->ptr) {
            
            // Merge blocks
            current->size += next->size;
            current->next = next->next;
            if (next->next) {
                next->next->prev = current;
            }
            
            delete next;
        } else {
            current = current->next;
        }
    }
}

void MemoryPool::expandPool(size_t min_size) {
    size_t new_size = std::max(min_size, config_.initial_size);
    pools_.emplace_back(std::make_unique<Pool>(new_size));
    stats_.pool_size += new_size;
}

size_t MemoryPool::alignSize(size_t size, size_t alignment) const {
    return (size + alignment - 1) & ~(alignment - 1);
}

void MemoryPool::updateStatistics(size_t size, bool is_allocation) {
    if (is_allocation) {
        stats_.total_allocated += size;
        stats_.current_usage += size;
        stats_.num_allocations++;
        stats_.peak_usage = std::max(stats_.peak_usage, stats_.current_usage);
    } else {
        stats_.total_deallocated += size;
        stats_.current_usage -= size;
        stats_.num_deallocations++;
    }
}

void* MemoryPool::alignPointer(void* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned);
}

size_t MemoryPool::getAlignmentOffset(void* ptr, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return aligned - addr;
}

// GlobalMemoryManager implementation
GlobalMemoryManager& GlobalMemoryManager::getInstance() {
    static GlobalMemoryManager instance;
    return instance;
}

GlobalMemoryManager::GlobalMemoryManager() : peak_memory_usage_(0) {
    // Initialize default pools
    MemoryPool::Config tensor_config;
    tensor_config.initial_size = 64 * 1024 * 1024;  // 64MB for tensors
    tensor_config.max_size = 2ULL * 1024 * 1024 * 1024; // 2GB max
    
    MemoryPool::Config workspace_config;
    workspace_config.initial_size = 32 * 1024 * 1024;  // 32MB for workspace
    workspace_config.max_size = 1ULL * 1024 * 1024 * 1024; // 1GB max
    
    MemoryPool::Config cache_config;
    cache_config.initial_size = 128 * 1024 * 1024; // 128MB for KV cache
    cache_config.max_size = 4ULL * 1024 * 1024 * 1024; // 4GB max
    
    tensor_pool_ = std::make_unique<MemoryPool>(tensor_config);
    workspace_pool_ = std::make_unique<MemoryPool>(workspace_config);
    kv_cache_pool_ = std::make_unique<MemoryPool>(cache_config);
    general_pool_ = std::make_unique<MemoryPool>(16 * 1024 * 1024); // 16MB general
}

GlobalMemoryManager::~GlobalMemoryManager() = default;

MemoryPool* GlobalMemoryManager::getTensorPool() {
    return tensor_pool_.get();
}

MemoryPool* GlobalMemoryManager::getWorkspacePool() {
    return workspace_pool_.get();
}

MemoryPool* GlobalMemoryManager::getKVCachePool() {
    return kv_cache_pool_.get();
}

MemoryPool* GlobalMemoryManager::getGeneralPool() {
    return general_pool_.get();
}

void GlobalMemoryManager::configurePools(const MemoryPool::Config& tensor_config,
                                        const MemoryPool::Config& workspace_config,
                                        const MemoryPool::Config& cache_config) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    tensor_pool_ = std::make_unique<MemoryPool>(tensor_config);
    workspace_pool_ = std::make_unique<MemoryPool>(workspace_config);
    kv_cache_pool_ = std::make_unique<MemoryPool>(cache_config);
}

GlobalMemoryManager::GlobalStats GlobalMemoryManager::getGlobalStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    GlobalStats stats;
    stats.tensor_stats = tensor_pool_->getStatistics();
    stats.workspace_stats = workspace_pool_->getStatistics();
    stats.cache_stats = kv_cache_pool_->getStatistics();
    stats.general_stats = general_pool_->getStatistics();
    
    stats.total_memory_usage = stats.tensor_stats.current_usage +
                              stats.workspace_stats.current_usage +
                              stats.cache_stats.current_usage +
                              stats.general_stats.current_usage;
    
    stats.peak_memory_usage = peak_memory_usage_;
    peak_memory_usage_ = std::max(peak_memory_usage_, stats.total_memory_usage);
    
    return stats;
}

void GlobalMemoryManager::resetAllStatistics() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    tensor_pool_->resetStatistics();
    workspace_pool_->resetStatistics();
    kv_cache_pool_->resetStatistics();
    general_pool_->resetStatistics();
    peak_memory_usage_ = 0;
}

void GlobalMemoryManager::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    workspace_pool_->clear();
    general_pool_->clear();
    // Don't clear tensor and cache pools as they may contain persistent data
}

void GlobalMemoryManager::defragmentAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    tensor_pool_->defragment();
    workspace_pool_->defragment();
    kv_cache_pool_->defragment();
    general_pool_->defragment();
}

} // namespace deepcpp 