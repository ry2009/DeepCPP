# Utils CMakeLists.txt

add_subdirectory(profiling)
add_subdirectory(visualization)
add_subdirectory(serialization)

# Create placeholder source files
file(TOUCH profiling/timer.cpp)
file(TOUCH profiling/memory_tracker.cpp)
file(TOUCH visualization/attention_maps.cpp)
file(TOUCH serialization/model_io.cpp) 