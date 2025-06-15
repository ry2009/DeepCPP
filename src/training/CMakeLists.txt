# Training CMakeLists.txt

# Training library
add_library(deepcpp_training STATIC
    custom_trainer.cpp
)

target_include_directories(deepcpp_training PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(deepcpp_training PUBLIC
    deepcpp_core
    deepcpp_operators
    deepcpp_attention
    deepcpp_models
)

# Link OpenMP if available
if(APPLE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(deepcpp_training PUBLIC OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(deepcpp_training PUBLIC "-L/usr/local/opt/libomp/lib -lomp")
    endif()
endif()

# Compiler-specific optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(deepcpp_training PRIVATE -O3 -march=native -ffast-math)
endif()

# Add filesystem library for C++17 compatibility
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(deepcpp_training PUBLIC stdc++fs)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(deepcpp_training PUBLIC c++fs)
endif() 