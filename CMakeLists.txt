cmake_minimum_required(VERSION 3.20)
project(DeepCppFramework VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /arch:AVX2")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
endif()

# Options for the massive framework
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_OPENCL "Enable OpenCL support" OFF)
option(USE_MKL "Use Intel MKL for BLAS operations" OFF)
option(USE_OPENBLAS "Use OpenBLAS for BLAS operations" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_BENCHMARKS "Build comprehensive benchmarks" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build example applications" ON)
option(USE_PROFILING "Enable performance profiling" OFF)

# Find required packages
find_package(PkgConfig REQUIRED)

# OpenMP support for parallel processing
if(APPLE)
    # Manual libomp configuration for Apple clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/opt/libomp/lib -lomp")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/usr/local/opt/libomp/lib -lomp")
    
    # Also find OpenMP for target-based linking
    find_package(OpenMP)
else()
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

# CUDA support
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -O3")
    add_definitions(-DUSE_CUDA)
endif()

# OpenCL support
if(USE_OPENCL)
    find_package(OpenCL REQUIRED)
    add_definitions(-DUSE_OPENCL)
endif()

# BLAS libraries (optional for now)
if(USE_MKL)
    find_package(MKL QUIET)
    if(MKL_FOUND)
        add_definitions(-DUSE_MKL)
    endif()
elseif(USE_OPENBLAS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENBLAS QUIET openblas)
    if(OPENBLAS_FOUND)
        add_definitions(-DUSE_OPENBLAS)
    endif()
endif()

# ONNX Runtime
if(APPLE)
    # Homebrew paths for macOS
    set(ONNXRUNTIME_ROOT_PATH /usr/local/Cellar/onnxruntime/1.22.0)
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime)
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.dylib)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ONNXRUNTIME REQUIRED libonnxruntime)
endif()

# Protobuf for model serialization
find_package(Protobuf REQUIRED)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, skipping Python bindings")
endif()
endif()

# Google Test for unit testing
if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
endif()

# Google Benchmark for performance testing
if(BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found, skipping benchmarks")
    set(BUILD_BENCHMARKS OFF)
endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
)

# Add subdirectories for our massive framework
add_subdirectory(src/core)
add_subdirectory(src/operators)  
# add_subdirectory(src/models)
add_subdirectory(src/training)
# add_subdirectory(src/inference)
# add_subdirectory(src/utils)

# Add benchmarks and integration
if(BUILD_BENCHMARKS)
    add_subdirectory(src/benchmarks)
endif()
# Temporarily disable integration due to ONNX API compatibility issues
# # add_subdirectory(src/integration)

# ---------- Custom kernels library ----------
add_library(custom_ops SHARED
  src/custom_ops.cc
  src/kernels/ssm_scan.cpp
)

target_include_directories(custom_ops PRIVATE
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(custom_ops PRIVATE ${ONNXRUNTIME_LIBRARIES})

# Link OpenMP to custom_ops if found
if(APPLE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(custom_ops PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(custom_ops PRIVATE "-L/usr/local/opt/libomp/lib -lomp")
    endif()
endif()

# ---------- Main inference application ----------
add_executable(run_infer src/main.cpp)

target_include_directories(run_infer PRIVATE
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(run_infer PRIVATE 
  ${ONNXRUNTIME_LIBRARIES}
  custom_ops
  deepcpp_core
  deepcpp_operators
)

# Link OpenMP to main app if found
if(APPLE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(run_infer PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(run_infer PRIVATE "-L/usr/local/opt/libomp/lib -lomp")
    endif()
endif()

# ---------- Benchmark application ----------
add_executable(benchmark benchmark.cpp)

target_include_directories(benchmark PRIVATE
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(benchmark PRIVATE 
  ${ONNXRUNTIME_LIBRARIES}
)

# Link OpenMP to benchmark if found
if(APPLE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(benchmark PRIVATE "-L/usr/local/opt/libomp/lib -lomp")
    endif()
endif()

# ---------- Comprehensive Benchmark Application ----------
if(BUILD_BENCHMARKS)
    add_executable(comprehensive_benchmark comprehensive_benchmark_app.cpp)
    
    target_include_directories(comprehensive_benchmark PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    
    target_link_libraries(comprehensive_benchmark PRIVATE 
        ${ONNXRUNTIME_LIBRARIES}
        deepcpp_core
        deepcpp_operators
        deepcpp_attention
        deepcpp_models
        deepcpp_performance
        deepcpp_benchmarks
        # deepcpp_integration
    )
    
    # Link OpenMP to comprehensive benchmark if found
    if(APPLE)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(comprehensive_benchmark PRIVATE OpenMP::OpenMP_CXX)
        else()
            target_link_libraries(comprehensive_benchmark PRIVATE "-L/usr/local/opt/libomp/lib -lomp")
        endif()
    endif()
endif()

# ---------- Simple Component Benchmark Application ----------
add_executable(simple_benchmark simple_component_benchmark.cpp)

target_include_directories(simple_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(simple_benchmark PRIVATE 
    deepcpp_core
    deepcpp_operators
    deepcpp_attention
    deepcpp_models
    deepcpp_performance
)

# Link OpenMP to simple benchmark if found
if(APPLE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(simple_benchmark PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(simple_benchmark PRIVATE "-L/usr/local/opt/libomp/lib -lomp")
    endif()
endif()

# Performance optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(run_infer PRIVATE -O3 -march=native)
  target_compile_options(custom_ops PRIVATE -O3 -march=native)
  
  # Strip symbols for smaller binaries
  if(NOT APPLE)
    set_target_properties(run_infer PROPERTIES LINK_FLAGS "-s")
    set_target_properties(custom_ops PROPERTIES LINK_FLAGS "-s")
  endif()
endif()

# Copy models to build directory for easy testing
file(GLOB MODEL_FILES "${CMAKE_SOURCE_DIR}/models/*.onnx")
if(MODEL_FILES)
  file(COPY ${MODEL_FILES} DESTINATION ${CMAKE_BINARY_DIR}/models/)
endif()

# Add test for real capabilities
add_executable(test_real_capabilities test_real_capabilities.cpp)
target_link_libraries(test_real_capabilities 
    deepcpp_attention 
    deepcpp_models 
    deepcpp_tensor 
    deepcpp_performance 
    deepcpp_activations 
    deepcpp_normalization 
    deepcpp_memory
)

# ---------- Custom Training Demo Application ----------
add_executable(custom_training_demo custom_training_demo.cpp)

target_include_directories(custom_training_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(custom_training_demo PRIVATE 
    deepcpp_core
    deepcpp_operators
    deepcpp_attention
    deepcpp_models
    deepcpp_training
)

# Link OpenMP to training demo if found
if(APPLE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(custom_training_demo PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(custom_training_demo PRIVATE "-L/usr/local/opt/libomp/lib -lomp")
    endif()
endif() 