add_library(ggml STATIC ggml.c ggml.h)
target_compile_features(ggml PUBLIC c_std_11)
target_link_libraries(ggml PRIVATE Threads::Threads)  # TODO: Use Accelerate.

add_library(utils STATIC utils.cc utils.h)
target_include_directories(utils PUBLIC .)
target_compile_features(utils PUBLIC cxx_std_11)

add_executable(llama main.cc)
target_link_libraries(llama PRIVATE ggml utils)

add_executable(quantize quantize.cc)
target_link_libraries(quantize PRIVATE ggml utils)
