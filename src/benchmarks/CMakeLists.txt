# Benchmarks library
add_library(deepcpp_benchmarks STATIC
    comprehensive_benchmark.cpp
)

target_include_directories(deepcpp_benchmarks PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(deepcpp_benchmarks PUBLIC
    deepcpp_core
    deepcpp_operators
    deepcpp_attention
    deepcpp_models
    deepcpp_performance
)

# Link OpenMP if available
if(APPLE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(deepcpp_benchmarks PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(deepcpp_benchmarks PRIVATE "-L/usr/local/opt/libomp/lib -lomp")
    endif()
endif() 