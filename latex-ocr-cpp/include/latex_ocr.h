#pragma once

#include <string>
#include <vector>
#include <memory>

namespace latex_ocr {

struct Config {
    // Model architecture
    int dim = 256;
    int encoder_depth = 4;
    int decoder_depth = 4;
    int heads = 8;
    int vocab_size = 8000;
    int max_seq_len = 512;
    int patch_size = 16;
    int channels = 1;
    int max_width = 672;
    int max_height = 192;
    int backbone_layers[3] = {2, 3, 7};

    // Decoder features
    bool attn_on_attn = true;
    bool ff_glu = true;
    bool cross_attend = true;

    // Special tokens
    int pad_token = 0;
    int bos_token = 1;
    int eos_token = 2;

    // Inference
    float temperature = 0.2f;
    int top_k = 40;
    bool use_gpu = true;
    int n_threads = 4;
};

class LatexOCR {
public:
    LatexOCR(const std::string& model_path, const std::string& tokenizer_path);
    ~LatexOCR();

    // Disable copy
    LatexOCR(const LatexOCR&) = delete;
    LatexOCR& operator=(const LatexOCR&) = delete;

    // Move is okay
    LatexOCR(LatexOCR&&) noexcept;
    LatexOCR& operator=(LatexOCR&&) noexcept;

    // Recognize LaTeX from image file
    std::string recognize_file(const std::string& image_path);

    // Recognize LaTeX from raw image data (RGB or grayscale)
    std::string recognize(const uint8_t* image_data, int width, int height, int channels);

    // Configuration
    void set_temperature(float temp);
    void set_max_tokens(int max_tokens);
    void set_use_gpu(bool use_gpu);
    void set_n_threads(int n);

    // Info
    Config get_config() const;
    size_t model_size_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace latex_ocr
