#pragma once

#include "latex_ocr.h"
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace latex_ocr {

struct ModelWeights;

// KV Cache for autoregressive decoding
struct KVCache {
    struct ggml_tensor* k;  // [layers, heads, max_seq_len, head_dim]
    struct ggml_tensor* v;  // [layers, heads, max_seq_len, head_dim]
    int current_len;
    int max_len;

    void reset() { current_len = 0; }
};

// Build decoder graph for single token prediction (used in generation)
// token_id: current token to process
// encoder_output: [batch, enc_seq_len, dim]
// Returns logits: [vocab_size]
struct ggml_tensor* build_decoder_step(
    struct ggml_context* ctx,
    struct ggml_tensor* token_embedding,
    struct ggml_tensor* encoder_output,
    const ModelWeights& weights,
    KVCache& self_kv_cache,
    const Config& config,
    int position
);

// Build single decoder transformer block
struct ggml_tensor* build_decoder_block(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* encoder_output,
    const ModelWeights& weights,
    int block_idx,
    KVCache& self_kv_cache,
    const Config& config,
    int position
);

// Build cross-attention (decoder attends to encoder output)
struct ggml_tensor* build_cross_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* context,
    struct ggml_tensor* wq,
    struct ggml_tensor* wk,
    struct ggml_tensor* wv,
    struct ggml_tensor* wo,
    int heads,
    int head_dim
);

// Build causal self-attention with KV cache
struct ggml_tensor* build_causal_self_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* wq,
    struct ggml_tensor* wk,
    struct ggml_tensor* wv,
    struct ggml_tensor* wo,
    KVCache& kv_cache,
    int layer_idx,
    int heads,
    int head_dim,
    int position
);

// Build FFN with GLU (Gated Linear Unit)
struct ggml_tensor* build_ffn_glu(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w1,
    struct ggml_tensor* w2,
    struct ggml_tensor* w_gate
);

// Autoregressive generation
std::vector<int> generate(
    struct ggml_context* ctx,
    struct ggml_tensor* encoder_output,
    const ModelWeights& weights,
    const Config& config,
    float temperature,
    int max_tokens,
    int bos_token,
    int eos_token
);

// Top-k sampling from logits
int sample_top_k(const float* logits, int vocab_size, float temperature, int top_k);

} // namespace latex_ocr
