
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
add_executable(demo_app ../demo_app.cpp)
target_link_libraries(demo_app deepcpp_attention deepcpp_models deepcpp_tensor deepcpp_memory)
if(APPLE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(demo_app PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(demo_app PRIVATE "-L/usr/local/opt/libomp/lib -lomp")
    endif()
endif()
add_executable(test_long_sequences ../test_long_sequences.cpp)
target_link_libraries(test_long_sequences deepcpp_attention deepcpp_tensor deepcpp_memory)
if(APPLE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_long_sequences PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(test_long_sequences PRIVATE "-L/usr/local/opt/libomp/lib -lomp")
    endif()
endif()
