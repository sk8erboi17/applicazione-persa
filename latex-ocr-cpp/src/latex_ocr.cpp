#include "latex_ocr.h"
#include "tokenizer.h"
#include "image_preprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>

namespace latex_ocr {

struct LatexOCR::Impl {
    ModelState* model = nullptr;
    Tokenizer tokenizer;
    Config config;
    float temperature = 0.2f;
    int max_tokens = 512;
};

LatexOCR::LatexOCR(const std::string& model_path, const std::string& tokenizer_path)
    : pimpl(std::make_unique<Impl>())
{
    // Load tokenizer
    if (!pimpl->tokenizer.load(tokenizer_path)) {
        throw std::runtime_error("Failed to load tokenizer: " + tokenizer_path);
    }

    // Load model
    pimpl->model = load_model(model_path, true);
    if (!pimpl->model) {
        throw std::runtime_error("Failed to load model: " + model_path);
    }

    pimpl->config = pimpl->model->config;
    pimpl->temperature = pimpl->config.temperature;

    print_model_info(pimpl->model);
}

LatexOCR::~LatexOCR() {
    if (pimpl && pimpl->model) {
        free_model(pimpl->model);
    }
}

LatexOCR::LatexOCR(LatexOCR&&) noexcept = default;
LatexOCR& LatexOCR::operator=(LatexOCR&&) noexcept = default;

std::string LatexOCR::recognize_file(const std::string& image_path) {
    auto start = std::chrono::high_resolution_clock::now();

    // Load and preprocess image
    ImageData img = load_and_preprocess(
        image_path,
        pimpl->config.max_width,
        pimpl->config.max_height
    );

    if (img.pixels.empty()) {
        return "";
    }

    auto preprocess_end = std::chrono::high_resolution_clock::now();

    // Run inference
    std::vector<int> token_ids = run_inference(
        pimpl->model,
        img.pixels.data(),
        img.width,
        img.height,
        pimpl->temperature,
        pimpl->max_tokens
    );

    auto inference_end = std::chrono::high_resolution_clock::now();

    // Decode tokens to LaTeX string
    std::string result = pimpl->tokenizer.decode(token_ids);

    auto total_end = std::chrono::high_resolution_clock::now();

    // Report timing
    auto preprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_end - start).count();
    auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - preprocess_end).count();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - start).count();

    std::cout << "Timing: preprocess=" << preprocess_ms << "ms"
              << " inference=" << inference_ms << "ms"
              << " total=" << total_ms << "ms"
              << " tokens=" << token_ids.size() << std::endl;

    return result;
}

std::string LatexOCR::recognize(const uint8_t* image_data, int width, int height, int channels) {
    // Preprocess
    ImageData img = preprocess(
        image_data, width, height, channels,
        pimpl->config.max_width,
        pimpl->config.max_height
    );

    if (img.pixels.empty()) return "";

    // Run inference
    std::vector<int> token_ids = run_inference(
        pimpl->model,
        img.pixels.data(),
        img.width,
        img.height,
        pimpl->temperature,
        pimpl->max_tokens
    );

    return pimpl->tokenizer.decode(token_ids);
}

void LatexOCR::set_temperature(float temp) {
    pimpl->temperature = temp;
}

void LatexOCR::set_max_tokens(int max_tokens) {
    pimpl->max_tokens = max_tokens;
}

void LatexOCR::set_use_gpu(bool use_gpu) {
    pimpl->config.use_gpu = use_gpu;
}

void LatexOCR::set_n_threads(int n) {
    pimpl->config.n_threads = n;
}

Config LatexOCR::get_config() const {
    return pimpl->config;
}

size_t LatexOCR::model_size_bytes() const {
    return pimpl->model ? pimpl->model->model_size_bytes : 0;
}

} // namespace latex_ocr
