#pragma once

#include "latex_ocr.h"

struct ggml_context;
struct ggml_tensor;

namespace latex_ocr {

// Forward declaration
struct ModelWeights;

// Build the encoder computation graph with ggml
// Input: images tensor [batch, channels, height, width]
// Output: encoded tensor [batch, num_patches+1, dim]
struct ggml_tensor* build_encoder_graph(
    struct ggml_context* ctx,
    struct ggml_tensor* images,
    const ModelWeights& weights,
    const Config& config,
    int actual_height,
    int actual_width
);

// Build ResNetV2 backbone subgraph
struct ggml_tensor* build_resnet_backbone(
    struct ggml_context* ctx,
    struct ggml_tensor* input,
    const ModelWeights& weights,
    const Config& config
);

// Build single encoder transformer block
struct ggml_tensor* build_encoder_block(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    const ModelWeights& weights,
    int block_idx,
    const Config& config
);

// Build self-attention
struct ggml_tensor* build_self_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* wq,
    struct ggml_tensor* wk,
    struct ggml_tensor* wv,
    struct ggml_tensor* wo,
    int heads,
    int head_dim
);

} // namespace latex_ocr
