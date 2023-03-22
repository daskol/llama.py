#pragma once

#include <llama/cc/ggml.h>
#include <llama/cc/utils.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Default hparams for LLaMA 7B.
struct llama_hparams {
    int32_t n_vocab = 32000;
    int32_t n_ctx = 512; // Is this provided as user input?
    int32_t n_embd = 4096;
    int32_t n_mult = 256;
    int32_t n_head = 32;
    int32_t n_layer = 32;
    int32_t n_rot = 64;
    int32_t f16 = 1;
};

struct llama_layer {
    // normalization
    struct ggml_tensor *attention_norm;

    // attention
    struct ggml_tensor *wq;
    struct ggml_tensor *wk;
    struct ggml_tensor *wv;
    struct ggml_tensor *wo;

    // normalization
    struct ggml_tensor *ffn_norm;

    // ff
    struct ggml_tensor *w1;
    struct ggml_tensor *w2;
    struct ggml_tensor *w3;
};

// Forward declaration for llama_model.
struct llama_model {
    llama_hparams hparams;

    struct ggml_tensor *tok_embeddings;

    struct ggml_tensor *norm;
    struct ggml_tensor *output;

    std::vector<llama_layer> layers;

    // key + value memory
    struct ggml_tensor *memory_k;
    struct ggml_tensor *memory_v;

    //
    struct ggml_context *ctx;
    std::unordered_map<std::string, struct ggml_tensor *> tensors;
};

namespace llama {

using DType = ggml_type; //< Alias for verbosity.

class Tokenizer {
public:
    using ID = llama_vocab::id;

private:
    llama_vocab vocab_;

public:
    Tokenizer(llama_vocab &&vocab) : vocab_{std::move(vocab)} {
    }

    std::string Decode(ID id) const {
        return vocab_.id_to_token.at(id).tok;
    }

    /**
     * Encode text to sequence of token identifiers.
     *
     * @param[in] bos Add Begin of Sequence (BoS) token or not.
     * @return Sequence of token ids.
     */
    std::vector<ID> Encode(std::string const &text, bool bos = true);

    llama_vocab const &GetVocab(void) const {
        return vocab_;
    }

    static std::shared_ptr<Tokenizer> Load(std::string const &path);
};

class LLaMA {
private:
    std::unique_ptr<llama_model> model_;
    std::shared_ptr<Tokenizer> tokenizer_;

public:
    LLaMA(std::unique_ptr<llama_model> &&model,
          std::shared_ptr<Tokenizer> tokenizer_)
        : model_{std::move(model)}, tokenizer_{tokenizer_} {
    }

    virtual ~LLaMA(void);

    /**
     * Apply model to input tokens and calculate logits of the next token.
     *
     * @param[in] context           Context tokens.
     * @param[in] context_size      Size of past context.
     * @param[out] logits           Logits of the next predicted token.
     * @param[in,out] mem_per_token Memory estimation needed for inference.
     * @param[in] nothreads         Number of threads to use.
     * @param[in] return_all_logits Return all logits.
     * @return Status of successfull computations.
     */
    bool Apply(std::vector<Tokenizer::ID> const &context, size_t context_size,
               std::vector<float> &logits, size_t &mem_per_token,
               size_t nothreads = 1, bool return_all_logits = false);

    void CalcPerplexity(std::string const &text, size_t context_size,
                        size_t mem_per_token, size_t nothreads = 1);

    size_t EstimateMemPerToken(size_t nothreads = 1);

    std::vector<float> Eval(std::vector<Tokenizer::ID> const &context,
                            size_t context_size, size_t mem_per_token,
                            size_t nothreads = 1,
                            bool return_all_logits = false);

    llama_hparams GetHParams(void) const {
        return model_->hparams;
    }

    std::shared_ptr<Tokenizer> GetTokenizer(void) {
        return tokenizer_;
    }

    static std::shared_ptr<LLaMA> Load(std::string const &path,
                                       size_t context_size,
                                       DType dtype = ggml_type::GGML_TYPE_F32);
};

/**
 * Sample next token with given probabilities for each embedding. Sampling
 * procedure is two step: (1) consider only the top K tokens; (2) from them,
 * consider only the top tokens with cumulative probability greater P.
 */
Tokenizer::ID SampleNextToken(Tokenizer const &tokenizer, float const *logits,
                              std::vector<llama_vocab::id> &last_n_tokens,
                              double repeat_penalty, int top_k, double top_p,
                              double temp, std::mt19937 &rng);

} // namespace llama
