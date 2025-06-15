# Operators library components
cmake_minimum_required(VERSION 3.16)

# Attention variants
add_library(deepcpp_attention STATIC
    attention/flash_attention.cpp
    attention/sparse_attention.cpp
    attention/linear_attention.cpp
    attention/multi_query_attention.cpp
)

target_include_directories(deepcpp_attention PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(deepcpp_attention PRIVATE
    deepcpp_core
)

# OpenMP support for attention
if(OpenMP_CXX_FOUND)
    target_link_libraries(deepcpp_attention PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(deepcpp_attention PRIVATE USE_OPENMP)
endif()

# Model architectures
add_library(deepcpp_models STATIC
    models/ssm.cpp
    models/mixture_of_experts.cpp
    models/transformer_blocks.cpp
)

target_include_directories(deepcpp_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(deepcpp_models PRIVATE
    deepcpp_core
)

# Activations
add_library(deepcpp_activations STATIC
    activations/gelu.cpp
    activations/swish.cpp
    activations/mish.cpp
    activations/geglu.cpp
)

target_include_directories(deepcpp_activations PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(deepcpp_activations PRIVATE
    deepcpp_core
)

# Normalization layers
add_library(deepcpp_normalization STATIC
    normalization/layer_norm.cpp
    normalization/rms_norm.cpp
    normalization/group_norm.cpp
    normalization/batch_norm.cpp
)

target_include_directories(deepcpp_normalization PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(deepcpp_normalization PRIVATE
    deepcpp_core
)

# Performance optimizations
add_library(deepcpp_performance STATIC
    performance/simd_kernels.cpp
    performance/kernel_fusion.cpp
    performance/memory_optimization.cpp
)

target_include_directories(deepcpp_performance PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(deepcpp_performance PRIVATE
    deepcpp_core
)

# Set common properties for all libraries
set(DEEPCPP_LIBRARIES 
    deepcpp_attention
    deepcpp_models
    deepcpp_activations
    deepcpp_normalization
    deepcpp_performance
)

# Apply common settings to all libraries
foreach(lib ${DEEPCPP_LIBRARIES})
    # AVX2/AVX-512 support
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(${lib} PRIVATE -mavx2 -mfma)
        if(ENABLE_AVX512)
            target_compile_options(${lib} PRIVATE -mavx512f -mavx512bw -mavx512vl)
            target_compile_definitions(${lib} PRIVATE USE_AVX512)
        endif()
        target_compile_definitions(${lib} PRIVATE USE_AVX2)
    endif()
    
    # C++ standard and optimization
    target_compile_features(${lib} PRIVATE cxx_std_17)
    target_compile_options(${lib} PRIVATE -O3 -march=native)
    
    # Debug information
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${lib} PRIVATE -g -O0)
        target_compile_definitions(${lib} PRIVATE DEBUG_MODE)
    endif()
endforeach()

# Create a combined operators library
add_library(deepcpp_operators INTERFACE)
target_link_libraries(deepcpp_operators INTERFACE ${DEEPCPP_LIBRARIES})

target_include_directories(deepcpp_operators INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Install targets
install(TARGETS ${DEEPCPP_LIBRARIES} deepcpp_operators
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY attention/ DESTINATION include/deepcpp/operators/attention
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY models/ DESTINATION include/deepcpp/operators/models
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY activations/ DESTINATION include/deepcpp/operators/activations
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY normalization/ DESTINATION include/deepcpp/operators/normalization
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY performance/ DESTINATION include/deepcpp/operators/performance
    FILES_MATCHING PATTERN "*.h")

# Placeholder files will be created as needed 