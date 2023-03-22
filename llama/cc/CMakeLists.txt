find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG)

add_library(ggml ggml.c ggml.h)
target_compile_features(ggml PUBLIC c_std_11)
target_link_libraries(ggml PRIVATE Threads::Threads)  # TODO: Use Accelerate.

add_library(utils utils.cc utils.h)
target_compile_features(utils PUBLIC cxx_std_11)

set_target_properties(ggml utils
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

add_executable(llama main.cc)
target_link_libraries(llama PRIVATE ggml utils)

add_executable(quantize quantization.h quantization.cc quantize.cc)
target_link_libraries(quantize PRIVATE ggml utils)

pybind11_add_module(_llama NO_EXTRAS
    module.cc
    quantization.h
    quantization.cc)
target_link_libraries(_llama PRIVATE ggml utils)
