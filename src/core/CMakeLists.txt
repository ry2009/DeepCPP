# Core components of the DeepCpp framework

# Memory management library
add_library(deepcpp_memory STATIC
    memory/memory_pool.cpp
)

target_include_directories(deepcpp_memory PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(deepcpp_memory PRIVATE
    ${CMAKE_THREAD_LIBS_INIT}
)

# Link OpenMP if available
if(APPLE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(deepcpp_memory PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(deepcpp_memory PRIVATE "-L/usr/local/opt/libomp/lib -lomp")
    endif()
endif()

# Tensor operations library (placeholder for now)
add_library(deepcpp_tensor STATIC
    tensor/tensor.cpp
)

target_include_directories(deepcpp_tensor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(deepcpp_tensor PRIVATE
    deepcpp_memory
)

# Core library combining all components
add_library(deepcpp_core INTERFACE)

target_link_libraries(deepcpp_core INTERFACE
    deepcpp_memory
    deepcpp_tensor
)

target_include_directories(deepcpp_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Placeholder files will be created as needed 