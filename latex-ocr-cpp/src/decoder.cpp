#include "decoder.h"
#include "model.h"

#include "ggml.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>

namespace latex_ocr {

// Helper: apply layer norm
static struct ggml_tensor* layer_norm(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma,
    struct ggml_tensor* beta
) {
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul(ctx, x, gamma);
    if (beta) x = ggml_add(ctx, x, beta);
    return x;
}

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
) {
    int64_t q_len = x->ne[1];
    int64_t kv_len = context->ne[1];
    int dim = heads * head_dim;

    // Q from decoder, K/V from encoder
    struct ggml_tensor* q = ggml_mul_mat(ctx, wq, x);      // [dim, q_len]
    struct ggml_tensor* k = ggml_mul_mat(ctx, wk, context); // [dim, kv_len]
    struct ggml_tensor* v = ggml_mul_mat(ctx, wv, context); // [dim, kv_len]

    // Reshape to multi-head
    q = ggml_reshape_3d(ctx, q, head_dim, heads, q_len);
    k = ggml_reshape_3d(ctx, k, head_dim, heads, kv_len);
    v = ggml_reshape_3d(ctx, v, head_dim, heads, kv_len);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // Attention
    struct ggml_tensor* attn = ggml_mul_mat(ctx, k, q);
    attn = ggml_scale(ctx, attn, 1.0f / sqrtf(static_cast<float>(head_dim)));
    attn = ggml_soft_max(ctx, attn);

    // Apply attention to values
    struct ggml_tensor* out = ggml_mul_mat(ctx, v, ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)));
    out = ggml_reshape_2d(ctx, ggml_cont(ctx, out), dim, q_len);

    // Output projection
    out = ggml_mul_mat(ctx, wo, out);

    return out;
}

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
) {
    (void)kv_cache;
    (void)layer_idx;
    (void)position;

    int64_t q_len = x->ne[1];
    int dim = heads * head_dim;

    struct ggml_tensor* q = ggml_mul_mat(ctx, wq, x);
    struct ggml_tensor* k = ggml_mul_mat(ctx, wk, x);
    struct ggml_tensor* v = ggml_mul_mat(ctx, wv, x);

    q = ggml_reshape_3d(ctx, q, head_dim, heads, q_len);
    k = ggml_reshape_3d(ctx, k, head_dim, heads, q_len);
    v = ggml_reshape_3d(ctx, v, head_dim, heads, q_len);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // Causal attention with mask
    struct ggml_tensor* attn = ggml_mul_mat(ctx, k, q);
    attn = ggml_scale(ctx, attn, 1.0f / sqrtf(static_cast<float>(head_dim)));

    // Apply causal mask (upper triangular -> -inf)
    attn = ggml_diag_mask_inf(ctx, attn, 0);
    attn = ggml_soft_max(ctx, attn);

    struct ggml_tensor* out = ggml_mul_mat(ctx, v, ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)));
    out = ggml_reshape_2d(ctx, ggml_cont(ctx, out), dim, q_len);
    out = ggml_mul_mat(ctx, wo, out);

    return out;
}

struct ggml_tensor* build_ffn_glu(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w1,
    struct ggml_tensor* w2,
    struct ggml_tensor* w_gate
) {
    // GLU: gate = sigmoid(x @ w_gate), out = (x @ w1) * gate
    struct ggml_tensor* hidden = ggml_mul_mat(ctx, w1, x);
    hidden = ggml_gelu(ctx, hidden);

    if (w_gate) {
        struct ggml_tensor* gate = ggml_mul_mat(ctx, w_gate, x);
        gate = ggml_sigmoid(ctx, gate);
        hidden = ggml_mul(ctx, hidden, gate);
    }

    struct ggml_tensor* out = ggml_mul_mat(ctx, w2, hidden);
    return out;
}

struct ggml_tensor* build_decoder_block(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* encoder_output,
    const ModelWeights& weights,
    int block_idx,
    KVCache& self_kv_cache,
    const Config& config,
    int position
) {
    char name[128];
    int head_dim = config.dim / config.heads;

    // 1. Pre-norm + Causal Self-Attention
    snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.norm.weight", block_idx * 3);
    struct ggml_tensor* ln1_w = weights.get(name);
    snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.norm.bias", block_idx * 3);
    struct ggml_tensor* ln1_b = weights.get(name);

    struct ggml_tensor* residual = x;
    if (ln1_w) x = layer_norm(ctx, x, ln1_w, ln1_b);

    snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.fn.to_q.weight", block_idx * 3);
    struct ggml_tensor* wq = weights.get(name);
    snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.fn.to_k.weight", block_idx * 3);
    struct ggml_tensor* wk = weights.get(name);
    snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.fn.to_v.weight", block_idx * 3);
    struct ggml_tensor* wv = weights.get(name);
    snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.fn.to_out.0.weight", block_idx * 3);
    struct ggml_tensor* wo = weights.get(name);

    if (wq && wk && wv && wo) {
        x = build_causal_self_attention(ctx, x, wq, wk, wv, wo,
                                         self_kv_cache, block_idx,
                                         config.heads, head_dim, position);
    }
    x = ggml_add(ctx, x, residual);

    // 2. Pre-norm + Cross-Attention (to encoder output)
    if (config.cross_attend && encoder_output) {
        snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.norm.weight", block_idx * 3 + 1);
        struct ggml_tensor* ln2_w = weights.get(name);
        snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.norm.bias", block_idx * 3 + 1);
        struct ggml_tensor* ln2_b = weights.get(name);

        residual = x;
        if (ln2_w) x = layer_norm(ctx, x, ln2_w, ln2_b);

        snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.fn.to_q.weight", block_idx * 3 + 1);
        struct ggml_tensor* cq = weights.get(name);
        snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.fn.to_k.weight", block_idx * 3 + 1);
        struct ggml_tensor* ck = weights.get(name);
        snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.fn.to_v.weight", block_idx * 3 + 1);
        struct ggml_tensor* cv = weights.get(name);
        snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.fn.to_out.0.weight", block_idx * 3 + 1);
        struct ggml_tensor* co = weights.get(name);

        if (cq && ck && cv && co) {
            x = build_cross_attention(ctx, x, encoder_output, cq, ck, cv, co,
                                       config.heads, head_dim);
        }
        x = ggml_add(ctx, x, residual);
    }

    // 3. Pre-norm + FFN (with optional GLU)
    int ffn_layer_idx = block_idx * 3 + 2;
    snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.norm.weight", ffn_layer_idx);
    struct ggml_tensor* ln3_w = weights.get(name);
    snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.norm.bias", ffn_layer_idx);
    struct ggml_tensor* ln3_b = weights.get(name);

    residual = x;
    if (ln3_w) x = layer_norm(ctx, x, ln3_w, ln3_b);

    snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.fn.net.0.0.weight", ffn_layer_idx);
    struct ggml_tensor* ff_w1 = weights.get(name);
    snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.fn.net.2.weight", ffn_layer_idx);
    struct ggml_tensor* ff_w2 = weights.get(name);

    // GLU gate weight (if ff_glu is enabled)
    struct ggml_tensor* ff_gate = nullptr;
    if (config.ff_glu) {
        snprintf(name, sizeof(name), "decoder.net.attn_layers.layers.%d.0.fn.fn.net.0.1.weight", ffn_layer_idx);
        ff_gate = weights.get(name);
    }

    if (ff_w1 && ff_w2) {
        x = build_ffn_glu(ctx, x, ff_w1, ff_w2, ff_gate);
    }
    x = ggml_add(ctx, x, residual);

    return x;
}

struct ggml_tensor* build_decoder_step(
    struct ggml_context* ctx,
    struct ggml_tensor* token_embedding,
    struct ggml_tensor* encoder_output,
    const ModelWeights& weights,
    KVCache& self_kv_cache,
    const Config& config,
    int position
) {
    struct ggml_tensor* x = token_embedding;

    // Add positional embedding
    struct ggml_tensor* pos_embed = weights.get("decoder.net.pos_emb.emb.weight");
    if (pos_embed) {
        struct ggml_tensor* pos = ggml_view_1d(ctx, pos_embed, config.dim,
                                                position * config.dim * sizeof(float));
        pos = ggml_reshape_2d(ctx, pos, config.dim, 1);
        x = ggml_add(ctx, x, pos);
    }

    // Decoder blocks
    for (int i = 0; i < config.decoder_depth; i++) {
        x = build_decoder_block(ctx, x, encoder_output, weights, i,
                                 self_kv_cache, config, position);
    }

    // Final norm
    struct ggml_tensor* norm_w = weights.get("decoder.net.norm.weight");
    struct ggml_tensor* norm_b = weights.get("decoder.net.norm.bias");
    if (norm_w) x = layer_norm(ctx, x, norm_w, norm_b);

    // Project to vocabulary
    struct ggml_tensor* lm_head = weights.get("decoder.net.to_logits.weight");
    if (lm_head) {
        x = ggml_mul_mat(ctx, lm_head, x);
    }

    return x;  // logits: [vocab_size, 1]
}

int sample_top_k(const float* logits, int vocab_size, float temperature, int top_k) {
    // Apply temperature
    std::vector<std::pair<float, int>> scored(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        scored[i] = {logits[i] / temperature, i};
    }

    // Partial sort for top-k
    if (top_k > 0 && top_k < vocab_size) {
        std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        scored.resize(top_k);
    }

    // Softmax over top-k
    float max_val = scored[0].first;
    float sum = 0.0f;
    for (auto& [score, idx] : scored) {
        score = expf(score - max_val);
        sum += score;
    }
    for (auto& [score, idx] : scored) {
        score /= sum;
    }

    // Multinomial sampling
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float cumsum = 0.0f;
    for (const auto& [prob, idx] : scored) {
        cumsum += prob;
        if (r <= cumsum) return idx;
    }

    return scored.back().second;
}

std::vector<int> generate(
    struct ggml_context* ctx,
    struct ggml_tensor* encoder_output,
    const ModelWeights& weights,
    const Config& config,
    float temperature,
    int max_tokens,
    int bos_token,
    int eos_token
) {
    (void)ctx;
    (void)encoder_output;
    (void)weights;

    std::vector<int> tokens;
    tokens.push_back(bos_token);

    KVCache self_kv_cache;
    self_kv_cache.k = nullptr;
    self_kv_cache.v = nullptr;
    self_kv_cache.current_len = 0;
    self_kv_cache.max_len = config.max_seq_len;

    for (int pos = 0; pos < max_tokens; pos++) {
        // Build and evaluate decoder step for current token
        // In actual implementation, this creates a ggml graph for each step
        // and evaluates it using the ggml backend

        // For now, this is a stub that demonstrates the interface.
        // The actual implementation requires:
        // 1. Creating a new ggml context for this step
        // 2. Looking up the token embedding
        // 3. Building the decoder graph
        // 4. Evaluating with ggml_backend_graph_compute
        // 5. Reading logits and sampling

        // Placeholder: would be filled by model.cpp integration
        int next_token = eos_token;  // Stub

        if (next_token == eos_token) break;
        tokens.push_back(next_token);

        self_kv_cache.current_len++;
    }

    return tokens;
}

} // namespace latex_ocr
