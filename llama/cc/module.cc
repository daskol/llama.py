#include <llama/cc/quantization.h>
#include <pybind11/pybind11.h>

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
        .export_values();

    m.def(
        "quantize_model", &llama::QuantizeModel,
        "Quantize checkpoint in GGLM format..\n"
        "\n"
        ":param src: Path to original checkpoint.\n"
        ":param dst: Path to quantized checkpoint.\n"
        ":param dtype: FP-type code: GGML_TYPE_Q4_0 (2), GGML_TYPE_Q4_1 (3).\n",
        py::arg("src"), py::arg("dst"), py::arg("dtype"));
}
