# Integration library
add_library(deepcpp_integration STATIC
    onnx_integration.cpp
)

target_include_directories(deepcpp_integration PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(deepcpp_integration PUBLIC
    deepcpp_core
    deepcpp_operators
    deepcpp_attention
    deepcpp_models
    deepcpp_performance
    ${ONNXRUNTIME_LIBRARIES}
)

# Link OpenMP if available
if(APPLE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(deepcpp_integration PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_link_libraries(deepcpp_integration PRIVATE "-L/usr/local/opt/libomp/lib -lomp")
    endif()
endif() 