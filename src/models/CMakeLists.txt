# Models CMakeLists.txt

add_subdirectory(transformers)
add_subdirectory(mamba)
add_subdirectory(retnet)
add_subdirectory(hybrid)

# Create placeholder source files for all architectures
file(TOUCH transformers/gpt.cpp)
file(TOUCH transformers/bert.cpp)
file(TOUCH transformers/llama.cpp)
file(TOUCH mamba/mamba_block.cpp)
file(TOUCH mamba/selective_ssm.cpp)
file(TOUCH retnet/retention.cpp)
file(TOUCH hybrid/mamba_transformer.cpp) 