# Inference CMakeLists.txt

add_subdirectory(engines)
add_subdirectory(generation)
add_subdirectory(caching)
add_subdirectory(batching)

# Create placeholder source files
file(TOUCH engines/transformer_engine.cpp)
file(TOUCH engines/mamba_engine.cpp)
file(TOUCH generation/beam_search.cpp)
file(TOUCH generation/sampling.cpp)
file(TOUCH caching/kv_cache.cpp)
file(TOUCH batching/dynamic_batching.cpp) 