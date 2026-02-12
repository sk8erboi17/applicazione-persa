#pragma once

#include "latex_ocr.h"
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;

namespace latex_ocr {

// Named tensor storage for model weights
struct ModelWeights {
    std::unordered_map<std::string, struct ggml_tensor*> tensors;

    struct ggml_tensor* get(const std::string& name) const {
        auto it = tensors.find(name);
        if (it != tensors.end()) return it->second;
        return nullptr;
    }
};

// Complete model state
struct ModelState {
    Config config;
    ModelWeights weights;

    struct ggml_context* ctx_weights = nullptr;   // Context for weight tensors
    struct ggml_backend* backend = nullptr;        // Compute backend (Metal or CPU)
    struct ggml_backend_buffer* buffer = nullptr;  // Weight buffer

    size_t total_params = 0;
    size_t model_size_bytes = 0;

    ~ModelState();
};

// Load GGUF model file
// Returns nullptr on failure
ModelState* load_model(const std::string& gguf_path, bool use_gpu = true);

// Free model resources
void free_model(ModelState* state);

// Run full inference: image -> token IDs
std::vector<int> run_inference(
    ModelState* model,
    const float* image_pixels,
    int image_width,
    int image_height,
    float temperature = 0.2f,
    int max_tokens = 512
);

// Print model info
void print_model_info(const ModelState* model);

} // namespace latex_ocr
