#include "encoder.h"
#include "model.h"

#include "ggml.h"

#include <cmath>
#include <cstring>
#include <iostream>

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
    x = ggml_add(ctx, x, beta);
    return x;
}

struct ggml_tensor* build_self_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* wq,
    struct ggml_tensor* wk,
    struct ggml_tensor* wv,
    struct ggml_tensor* wo,
    int heads,
    int head_dim
) {
    // x: [seq_len, dim]
    int64_t seq_len = x->ne[1];
    int64_t dim = x->ne[0];

    // Q, K, V projections
    struct ggml_tensor* q = ggml_mul_mat(ctx, wq, x);  // [seq_len, dim]
    struct ggml_tensor* k = ggml_mul_mat(ctx, wk, x);
    struct ggml_tensor* v = ggml_mul_mat(ctx, wv, x);

    // Reshape to [head_dim, heads, seq_len]
    q = ggml_reshape_3d(ctx, q, head_dim, heads, seq_len);
    k = ggml_reshape_3d(ctx, k, head_dim, heads, seq_len);
    v = ggml_reshape_3d(ctx, v, head_dim, heads, seq_len);

    // Permute to [head_dim, seq_len, heads]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // Attention: Q @ K^T / sqrt(d)
    struct ggml_tensor* attn = ggml_mul_mat(ctx, k, q);  // [seq_len, seq_len, heads]
    attn = ggml_scale(ctx, attn, 1.0f / sqrtf(static_cast<float>(head_dim)));
    attn = ggml_soft_max(ctx, attn);

    // Attention @ V
    struct ggml_tensor* out = ggml_mul_mat(ctx, v, ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)));

    // Reshape back to [dim, seq_len]
    out = ggml_reshape_2d(ctx, ggml_cont(ctx, out), dim, seq_len);

    // Output projection
    out = ggml_mul_mat(ctx, wo, out);

    return out;
}

struct ggml_tensor* build_encoder_block(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    const ModelWeights& weights,
    int block_idx,
    const Config& config
) {
    char name[128];
    int head_dim = config.dim / config.heads;

    // Pre-norm + Self Attention
    snprintf(name, sizeof(name), "encoder.blocks.%d.norm1.weight", block_idx);
    struct ggml_tensor* ln1_w = weights.get(name);
    snprintf(name, sizeof(name), "encoder.blocks.%d.norm1.bias", block_idx);
    struct ggml_tensor* ln1_b = weights.get(name);

    struct ggml_tensor* residual = x;
    x = layer_norm(ctx, x, ln1_w, ln1_b);

    // Self attention
    snprintf(name, sizeof(name), "encoder.blocks.%d.attn.qkv.weight", block_idx);
    struct ggml_tensor* qkv_w = weights.get(name);

    if (qkv_w) {
        // Combined QKV projection (timm style)
        struct ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, x);
        int64_t seq_len = x->ne[1];

        // Split into Q, K, V
        struct ggml_tensor* q = ggml_view_2d(ctx, qkv, config.dim, seq_len, qkv->nb[1], 0);
        struct ggml_tensor* k = ggml_view_2d(ctx, qkv, config.dim, seq_len, qkv->nb[1], config.dim * sizeof(float));
        struct ggml_tensor* v = ggml_view_2d(ctx, qkv, config.dim, seq_len, qkv->nb[1], 2 * config.dim * sizeof(float));

        // Reshape to multi-head
        q = ggml_reshape_3d(ctx, q, head_dim, config.heads, seq_len);
        k = ggml_reshape_3d(ctx, k, head_dim, config.heads, seq_len);
        v = ggml_reshape_3d(ctx, v, head_dim, config.heads, seq_len);

        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, v, 0, 2, 1, 3);

        struct ggml_tensor* attn = ggml_mul_mat(ctx, k, q);
        attn = ggml_scale(ctx, attn, 1.0f / sqrtf(static_cast<float>(head_dim)));
        attn = ggml_soft_max(ctx, attn);

        x = ggml_mul_mat(ctx, v, ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)));
        x = ggml_reshape_2d(ctx, ggml_cont(ctx, x), config.dim, seq_len);

        snprintf(name, sizeof(name), "encoder.blocks.%d.attn.proj.weight", block_idx);
        struct ggml_tensor* proj_w = weights.get(name);
        if (proj_w) x = ggml_mul_mat(ctx, proj_w, x);
    }

    x = ggml_add(ctx, x, residual);

    // Pre-norm + FFN
    snprintf(name, sizeof(name), "encoder.blocks.%d.norm2.weight", block_idx);
    struct ggml_tensor* ln2_w = weights.get(name);
    snprintf(name, sizeof(name), "encoder.blocks.%d.norm2.bias", block_idx);
    struct ggml_tensor* ln2_b = weights.get(name);

    residual = x;
    x = layer_norm(ctx, x, ln2_w, ln2_b);

    // FFN: fc1 -> GELU -> fc2
    snprintf(name, sizeof(name), "encoder.blocks.%d.mlp.fc1.weight", block_idx);
    struct ggml_tensor* fc1_w = weights.get(name);
    snprintf(name, sizeof(name), "encoder.blocks.%d.mlp.fc1.bias", block_idx);
    struct ggml_tensor* fc1_b = weights.get(name);
    snprintf(name, sizeof(name), "encoder.blocks.%d.mlp.fc2.weight", block_idx);
    struct ggml_tensor* fc2_w = weights.get(name);
    snprintf(name, sizeof(name), "encoder.blocks.%d.mlp.fc2.bias", block_idx);
    struct ggml_tensor* fc2_b = weights.get(name);

    if (fc1_w) {
        x = ggml_mul_mat(ctx, fc1_w, x);
        if (fc1_b) x = ggml_add(ctx, x, fc1_b);
        x = ggml_gelu(ctx, x);
        if (fc2_w) {
            x = ggml_mul_mat(ctx, fc2_w, x);
            if (fc2_b) x = ggml_add(ctx, x, fc2_b);
        }
    }

    x = ggml_add(ctx, x, residual);

    return x;
}

struct ggml_tensor* build_resnet_backbone(
    struct ggml_context* ctx,
    struct ggml_tensor* input,
    const ModelWeights& weights,
    const Config& config
) {
    // ResNetV2 backbone: stem -> 3 stages
    // For now, build a simplified version that matches the weight names

    struct ggml_tensor* x = input;

    // Stem: conv 7x7, stride 2
    struct ggml_tensor* stem_w = weights.get("encoder.patch_embed.backbone.stem.conv.weight");
    if (stem_w) {
        x = ggml_conv_2d(ctx, stem_w, x, 2, 2, 3, 3, 1, 1);
    }

    // The backbone processes through stages producing feature maps
    // which are then projected by HybridEmbed into patch embeddings
    // The exact layer names depend on the timm model serialization

    return x;
}

struct ggml_tensor* build_encoder_graph(
    struct ggml_context* ctx,
    struct ggml_tensor* images,
    const ModelWeights& weights,
    const Config& config,
    int actual_height,
    int actual_width
) {
    // 1. ResNet backbone (produces feature maps)
    struct ggml_tensor* x = build_resnet_backbone(ctx, images, weights, config);

    // 2. Patch embedding projection
    struct ggml_tensor* proj_w = weights.get("encoder.patch_embed.proj.weight");
    if (proj_w) {
        // Project backbone features to patch embeddings
        int ps = config.patch_size / 16;  // Adjusted patch size after backbone
        x = ggml_conv_2d(ctx, proj_w, x, ps, ps, 0, 0, 1, 1);
    }

    // Reshape from [channels, h, w] to [dim, num_patches]
    int64_t feat_h = x->ne[1];
    int64_t feat_w = x->ne[2];
    int num_patches = static_cast<int>(feat_h * feat_w);

    x = ggml_reshape_2d(ctx, ggml_cont(ctx, x), config.dim, num_patches);

    // 3. Prepend CLS token
    struct ggml_tensor* cls_token = weights.get("encoder.cls_token");
    if (cls_token) {
        cls_token = ggml_reshape_2d(ctx, cls_token, config.dim, 1);
        x = ggml_concat(ctx, cls_token, x, 1);  // [dim, num_patches+1]
    }

    // 4. Add positional embeddings (adaptive)
    struct ggml_tensor* pos_embed = weights.get("encoder.pos_embed");
    if (pos_embed) {
        // For adaptive pos embedding, we'd need to select the right positions
        // based on actual image dimensions. For now, add truncated pos embed.
        int64_t seq_len = x->ne[1];
        struct ggml_tensor* pos = ggml_view_2d(ctx, pos_embed, config.dim, seq_len,
                                                pos_embed->nb[1], 0);
        x = ggml_add(ctx, x, pos);
    }

    // 5. Transformer encoder blocks
    for (int i = 0; i < config.encoder_depth; i++) {
        x = build_encoder_block(ctx, x, weights, i, config);
    }

    // 6. Final layer norm
    struct ggml_tensor* norm_w = weights.get("encoder.norm.weight");
    struct ggml_tensor* norm_b = weights.get("encoder.norm.bias");
    if (norm_w && norm_b) {
        x = layer_norm(ctx, x, norm_w, norm_b);
    }

    return x;  // [dim, num_patches+1]
}

} // namespace latex_ocr
