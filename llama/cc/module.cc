#include <llama/cc/llama.h>
#include <llama/cc/quantization.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace {

void InitializeF16Tables(void) {
    struct ggml_init_params params = {0, nullptr};
    struct ggml_context *ctx = ggml_init(params);
    ggml_free(ctx);
}

} // namespace

namespace py = pybind11;

PYBIND11_MODULE(_llama, m) {
    InitializeF16Tables(); // This is pretty odd way to initialize global
                           // states.

    py::enum_<ggml_type>(m, "GGMLType")
        .value("Q4_0", ggml_type::GGML_TYPE_Q4_0)
        .value("Q4_1", ggml_type::GGML_TYPE_Q4_1)
        .value("I8", ggml_type::GGML_TYPE_I8)
        .value("I16", ggml_type::GGML_TYPE_I16)
        .value("I32", ggml_type::GGML_TYPE_I32)
        .value("F16", ggml_type::GGML_TYPE_F16)
        .value("F32", ggml_type::GGML_TYPE_F32)
        .export_values();

    py::class_<llama::Tokenizer>(m, "Tokenizer")
        .def("decode", &llama::Tokenizer::Decode)
        .def("encode", &llama::Tokenizer::Encode)
        .def_static("load", &llama::Tokenizer::Load);

    py::class_<llama::LLaMA>(m, "LLaMA")
        .def("calc_perplexity", &llama::LLaMA::CalcPerplexity)
        .def("estimate_mem_per_token", &llama::LLaMA::EstimateMemPerToken)
        .def("eval", &llama::LLaMA::Eval)
        .def("get_tokenizer", &llama::LLaMA::GetTokenizer)
        .def_static("load", &llama::LLaMA::Load);

    m.def("sample_next_token", &llama::SampleNextToken);

    m.def(
        "quantize_model", &llama::QuantizeModel,
        "Quantize checkpoint in GGLM format..\n"
        "\n"
        ":param src: Path to original checkpoint.\n"
        ":param dst: Path to quantized checkpoint.\n"
        ":param dtype: FP-type code: GGML_TYPE_Q4_0 (2), GGML_TYPE_Q4_1 (3).\n",
        py::arg("src"), py::arg("dst"), py::arg("dtype"));
}
