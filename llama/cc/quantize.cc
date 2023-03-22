#include <cstdio>
#include <string>

#include <llama/cc/ggml.h>
#include <llama/cc/quantization.h>

int main(int argc, char ** argv) {
    ggml_time_init();
    if (argc != 4) {
        fprintf(stderr, "usage: %s model-f32.bin model-quant.bin type\n", argv[0]);
        fprintf(stderr, "  type = 2 - q4_0\n");
        fprintf(stderr, "  type = 3 - q4_1\n");
        return 1;
    }

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    const int itype = atoi(argv[3]);

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        ggml_type dtype = GGML_TYPE_Q4_1;

        switch (itype) {
        case 2:
            dtype = GGML_TYPE_Q4_0;
            break;
        case 3:
            dtype = GGML_TYPE_Q4_1;
            break;
        default:
            fprintf(stderr, "%s: invalid quantization type %d\n", __func__, itype);
            return 1;
        }

        if (!llama::QuantizeModel(fname_inp, fname_out, dtype)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    return 0;
}
