#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

#include <llama/cc/ggml.h>
#include <llama/cc/llama.h>
#include <llama/cc/utils.h>

#if defined (_WIN32)
#pragma comment(lib,"kernel32.lib")
extern "C" __declspec(dllimport) void* __stdcall GetStdHandle(unsigned long nStdHandle);
extern "C" __declspec(dllimport) int __stdcall GetConsoleMode(void* hConsoleHandle, unsigned long* lpMode);
extern "C" __declspec(dllimport) int __stdcall SetConsoleMode(void* hConsoleHandle, unsigned long dwMode);
#endif

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

/* Keep track of current color of output, and emit ANSI code if it changes. */
enum console_state {
    CONSOLE_STATE_DEFAULT=0,
    CONSOLE_STATE_PROMPT,
    CONSOLE_STATE_USER_INPUT
}; 

static console_state con_st = CONSOLE_STATE_DEFAULT;
static bool con_use_color = false;

void set_console_state(console_state new_st)
{
    if (!con_use_color) return;
    // only emit color code if state changed
    if (new_st != con_st) {
        con_st = new_st;
        switch(con_st) {
        case CONSOLE_STATE_DEFAULT:
            printf(ANSI_COLOR_RESET);
            return;
        case CONSOLE_STATE_PROMPT:
            printf(ANSI_COLOR_YELLOW);
            return;
        case CONSOLE_STATE_USER_INPUT:
            printf(ANSI_BOLD ANSI_COLOR_GREEN);
            return;
        }
    }
}

static const int EOS_TOKEN_ID = 2;

static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    set_console_state(CONSOLE_STATE_DEFAULT);
    printf("\n"); // this also force flush stdout.
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            _exit(130);
        }
    }
}
#endif

const char * llama_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";

    return s.c_str();
}

int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;
    params.model = "models/llama-7B/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    con_use_color = params.use_color;

//    params.prompt = R"(// this function checks if the number n is prime
//bool is_prime(int n) {)";

    int64_t t_load_us = 0;

    std::shared_ptr<llama::LLaMA> model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        auto memory_type = params.memory_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        model = llama::LLaMA::Load(params.model, params.n_ctx, memory_type);
        if (!model) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }
        t_load_us = ggml_time_us() - t_start_us;
    }

    auto tokenizer = model->GetTokenizer();

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    std::vector<float> logits;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    model->Apply({0, 1, 2, 3}, 0, logits, mem_per_token, params.n_threads);

    if (params.perplexity) {
        model->CalcPerplexity(params.prompt, mem_per_token, params.n_ctx, params.n_threads);
        exit(0);
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');
    // tokenize the prompt
    std::vector<llama_vocab::id> embd_inp = tokenizer->Encode(params.prompt, true);

    params.n_predict = std::min(params.n_predict, model->GetHParams().n_ctx - (int) embd_inp.size());

    // prefix & suffix for instruct mode
    const std::vector<llama_vocab::id> inp_pfx = tokenizer->Encode("\n\n### Instruction:\n\n", true);
    const std::vector<llama_vocab::id> inp_sfx = tokenizer->Encode("\n\n### Response:\n\n", false);

    // in instruct mode, we inject a prefix and a suffix to each input by the user
    if (params.instruct) {
        params.interactive = true;
        params.antiprompt.push_back("### Instruction:\n\n");
    }

    // enable interactive mode if reverse prompt is specified
    if (params.antiprompt.size() != 0) {
        params.interactive = true;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    for (int i = 0; i < (int) embd_inp.size(); i++) {
        fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], tokenizer->Decode(embd_inp[i]).c_str());
    }
    fprintf(stderr, "\n");
    if (params.interactive) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        signal(SIGINT, sigint_handler);
#endif

        fprintf(stderr, "%s: interactive mode on.\n", __func__);

        if(params.antiprompt.size()) {
            for (auto antiprompt : params.antiprompt) {
                fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
            }
        }
    }
    fprintf(stderr, "sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    fprintf(stderr, "\n\n");

    std::vector<llama_vocab::id> embd;

    int last_n_size = params.repeat_last_n;
    std::vector<llama_vocab::id> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    if (params.interactive) {
        fprintf(stderr, "== Running in interactive mode. ==\n"
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
               " - Press Ctrl+C to interject at any time.\n"
#endif
               " - Press Return to return control to LLaMa.\n"
               " - If you want to submit another line, end your input in '\\'.\n\n");
        is_interacting = true;
    }

    int input_consumed = 0;
    bool input_noecho = false;

    int remaining_tokens = params.n_predict;

#if defined (_WIN32)
  if (params.use_color) {
        // Enable ANSI colors on Windows 10+
        unsigned long dwMode = 0;
        void* hConOut = GetStdHandle((unsigned long)-11); // STD_OUTPUT_HANDLE (-11)
        if (hConOut && hConOut != (void*)-1 && GetConsoleMode(hConOut, &dwMode) && !(dwMode & 0x4)) {
            SetConsoleMode(hConOut, dwMode | 0x4); // ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x4)
        }
    }
#endif
    // the first thing we will do is to output the prompt, so set color accordingly
    set_console_state(CONSOLE_STATE_PROMPT);

    while (remaining_tokens > 0 || params.interactive) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!model->Apply(embd, n_past, logits, mem_per_token, params.n_threads)) {
                fprintf(stderr, "Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if ((int) embd_inp.size() <= input_consumed) {
            // out of user input, sample next token
            const float top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;
            const float repeat_penalty = params.repeat_penalty;

            const int n_vocab = model->GetHParams().n_vocab;

            llama_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                if (params.ignore_eos) {
                    // set the logit of the eos token to zero to avoid sampling it
                    logits[logits.size() - n_vocab + EOS_TOKEN_ID] = 0;
                }

                id = llama::SampleNextToken(*tokenizer, logits.data() + (logits.size() - n_vocab), last_n_tokens, repeat_penalty, top_k, top_p, temp, rng);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);

            // echo this to console
            input_noecho = false;

            // decrement remaining sampling budget
            --remaining_tokens;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > input_consumed) {
                embd.push_back(embd_inp[input_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[input_consumed]);
                ++input_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (!input_noecho) {
            for (auto id : embd) {
                printf("%s", tokenizer->Decode(id).c_str());
            }
            fflush(stdout);
        }
        // reset color to default if we there is no pending user input
        if (!input_noecho && (int)embd_inp.size() == input_consumed) {
            set_console_state(CONSOLE_STATE_DEFAULT);
        }

        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        if (params.interactive && (int) embd_inp.size() <= input_consumed) {
            // check for reverse prompt
            std::string last_output;
            for (auto id : last_n_tokens) {
                last_output += tokenizer->Decode(id);
            }

            // Check if each of the reverse prompts appears at the end of the output.
            for (std::string antiprompt : params.antiprompt) {
                if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                    is_interacting = true;
                    break;
                }
            }
            if (is_interacting) {
                // potentially set color to indicate we are taking user input
                set_console_state(CONSOLE_STATE_USER_INPUT);

                if (params.instruct) {
                    input_consumed = embd_inp.size();
                    embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());

                    printf("\n> ");
                }

                std::string buffer;
                std::string line;
                bool another_line = true;
                do {
                    std::getline(std::cin, line);
                    if (line.empty() || line.back() != '\\') {
                        another_line = false;
                    } else {
                        line.pop_back(); // Remove the continue character
                    }
                    buffer += line + '\n'; // Append the line to the result
                } while (another_line);

                // done taking input, reset color
                set_console_state(CONSOLE_STATE_DEFAULT);

                std::vector<llama_vocab::id> line_inp = tokenizer->Encode(buffer, false);
                embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                if (params.instruct) {
                    embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                }

                remaining_tokens -= line_inp.size();

                input_noecho = true; // do not echo this again
            }
            is_interacting = false;
        }

        // end of text token
        if (embd.back() == EOS_TOKEN_ID) {
            if (params.interactive) {
                is_interacting = true;
            } else {
                fprintf(stderr, " [end of text]\n");
                break;
            }
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (params.interactive && remaining_tokens <= 0) {
            remaining_tokens = params.n_predict;
            is_interacting = true;
        }
    }

#if defined (_WIN32)
    signal(SIGINT, SIG_DFL);
#endif

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        fprintf(stderr, "\n\n");
        fprintf(stderr, "%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        fprintf(stderr, "%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        fprintf(stderr, "%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        fprintf(stderr, "%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        fprintf(stderr, "%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    set_console_state(CONSOLE_STATE_DEFAULT);

    return 0;
}
