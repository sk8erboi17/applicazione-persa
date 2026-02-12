#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace latex_ocr {

struct ImageData {
    std::vector<float> pixels;  // Normalized grayscale [H, W]
    int width;
    int height;
};

// Load image from file and preprocess for the model
// Returns grayscale, normalized, padded image
ImageData load_and_preprocess(const std::string& path,
                               int max_width = 672, int max_height = 192,
                               int divisible_by = 32);

// Preprocess raw image data
// image_data: raw pixel data (RGB or grayscale)
// channels: 1 for grayscale, 3 for RGB, 4 for RGBA
ImageData preprocess(const uint8_t* image_data, int width, int height, int channels,
                     int max_width = 672, int max_height = 192,
                     int divisible_by = 32);

// Normalize pixel values: (pixel / 255 - mean) / std
void normalize(float* pixels, int count, float mean = 0.7931f, float std = 0.1738f);

// Pad image to make dimensions divisible
ImageData pad_image(const float* pixels, int width, int height,
                    int target_width, int target_height,
                    float pad_value = 1.0f);

// Convert RGB to grayscale
void rgb_to_gray(const uint8_t* rgb, uint8_t* gray, int width, int height);

} // namespace latex_ocr
