#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>

namespace latex_ocr {

void rgb_to_gray(const uint8_t* rgb, uint8_t* gray, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        // ITU-R BT.601 luma coefficients
        float r = rgb[i * 3 + 0];
        float g = rgb[i * 3 + 1];
        float b = rgb[i * 3 + 2];
        gray[i] = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

void normalize(float* pixels, int count, float mean, float std) {
    for (int i = 0; i < count; i++) {
        pixels[i] = (pixels[i] - mean) / std;
    }
}

ImageData pad_image(const float* pixels, int width, int height,
                    int target_width, int target_height, float pad_value) {
    ImageData result;
    result.width = target_width;
    result.height = target_height;
    result.pixels.resize(target_width * target_height, pad_value);

    // Copy original pixels
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            result.pixels[y * target_width + x] = pixels[y * width + x];
        }
    }

    return result;
}

ImageData load_and_preprocess(const std::string& path,
                               int max_width, int max_height,
                               int divisible_by) {
    int w, h, channels;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load image: " << path << std::endl;
        return {};
    }

    ImageData result = preprocess(data, w, h, channels, max_width, max_height, divisible_by);
    stbi_image_free(data);
    return result;
}

ImageData preprocess(const uint8_t* image_data, int width, int height, int channels,
                     int max_width, int max_height, int divisible_by) {
    // Convert to grayscale
    std::vector<uint8_t> gray(width * height);

    if (channels == 1) {
        std::memcpy(gray.data(), image_data, width * height);
    } else if (channels == 3) {
        rgb_to_gray(image_data, gray.data(), width, height);
    } else if (channels == 4) {
        // RGBA -> RGB -> Gray
        std::vector<uint8_t> rgb(width * height * 3);
        for (int i = 0; i < width * height; i++) {
            uint8_t a = image_data[i * 4 + 3];
            // Blend with white background
            float alpha = a / 255.0f;
            rgb[i * 3 + 0] = static_cast<uint8_t>(image_data[i * 4 + 0] * alpha + 255.0f * (1.0f - alpha));
            rgb[i * 3 + 1] = static_cast<uint8_t>(image_data[i * 4 + 1] * alpha + 255.0f * (1.0f - alpha));
            rgb[i * 3 + 2] = static_cast<uint8_t>(image_data[i * 4 + 2] * alpha + 255.0f * (1.0f - alpha));
        }
        rgb_to_gray(rgb.data(), gray.data(), width, height);
    } else {
        std::cerr << "Unsupported channel count: " << channels << std::endl;
        return {};
    }

    // Resize if needed (simple bilinear)
    int new_w = width;
    int new_h = height;

    if (new_w > max_width || new_h > max_height) {
        float scale = std::min(
            static_cast<float>(max_width) / new_w,
            static_cast<float>(max_height) / new_h
        );
        new_w = static_cast<int>(new_w * scale);
        new_h = static_cast<int>(new_h * scale);

        // Bilinear resize
        std::vector<uint8_t> resized(new_w * new_h);
        for (int y = 0; y < new_h; y++) {
            for (int x = 0; x < new_w; x++) {
                float src_x = x * (static_cast<float>(width) / new_w);
                float src_y = y * (static_cast<float>(height) / new_h);

                int x0 = static_cast<int>(src_x);
                int y0 = static_cast<int>(src_y);
                int x1 = std::min(x0 + 1, width - 1);
                int y1 = std::min(y0 + 1, height - 1);

                float fx = src_x - x0;
                float fy = src_y - y0;

                float val =
                    gray[y0 * width + x0] * (1 - fx) * (1 - fy) +
                    gray[y0 * width + x1] * fx * (1 - fy) +
                    gray[y1 * width + x0] * (1 - fx) * fy +
                    gray[y1 * width + x1] * fx * fy;

                resized[y * new_w + x] = static_cast<uint8_t>(val);
            }
        }
        gray = std::move(resized);
        width = new_w;
        height = new_h;
    }

    // Pad to divisible dimensions
    int padded_w = ((width + divisible_by - 1) / divisible_by) * divisible_by;
    int padded_h = ((height + divisible_by - 1) / divisible_by) * divisible_by;
    padded_w = std::min(padded_w, max_width);
    padded_h = std::min(padded_h, max_height);

    // Convert to float and normalize
    ImageData result;
    result.width = padded_w;
    result.height = padded_h;
    result.pixels.resize(padded_w * padded_h);

    // Fill with white (1.0 before normalization -> becomes (1.0 - mean) / std)
    // We normalize raw pixel: pixel/255 first
    float norm_white = 1.0f;  // White pixel = 255/255 = 1.0

    // Fill padded area with white
    for (int i = 0; i < padded_w * padded_h; i++) {
        result.pixels[i] = norm_white;
    }

    // Copy and convert to [0, 1] range
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            result.pixels[y * padded_w + x] = gray[y * width + x] / 255.0f;
        }
    }

    // Normalize: (pixel - mean) / std
    normalize(result.pixels.data(), padded_w * padded_h, 0.7931f, 0.1738f);

    return result;
}

} // namespace latex_ocr
