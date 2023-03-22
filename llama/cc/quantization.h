#pragma once

#include <llama/cc/ggml.h>
#include <string>

namespace llama {

bool QuantizeModel(std::string const &fname_inp, std::string const &fname_out,
                   ggml_type dtype);

} // namespace llama
