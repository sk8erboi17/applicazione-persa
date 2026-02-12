#include "latex_ocr.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>

void print_usage(const char* program) {
    std::cout << "LaTeX-OCR C++ Inference Engine" << std::endl;
    std::cout << "Optimized for Apple Silicon M4" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << program << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -m, --model <path>      Path to GGUF model file (required)" << std::endl;
    std::cout << "  -t, --tokenizer <path>  Path to tokenizer.json (required)" << std::endl;
    std::cout << "  -i, --image <path>      Path to input image (required)" << std::endl;
    std::cout << "  --temperature <float>   Sampling temperature (default: 0.2)" << std::endl;
    std::cout << "  --max-tokens <int>      Maximum output tokens (default: 512)" << std::endl;
    std::cout << "  --cpu                   Force CPU-only (no Metal GPU)" << std::endl;
    std::cout << "  --threads <int>         Number of CPU threads (default: 4)" << std::endl;
    std::cout << "  -h, --help              Show this help" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program << " -m model.gguf -t tokenizer.json -i formula.png" << std::endl;
    std::cout << "  " << program << " -m model.gguf -t tokenizer.json -i photo.jpg --temperature 0.1" << std::endl;
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string tokenizer_path;
    std::vector<std::string> image_paths;
    float temperature = 0.2f;
    int max_tokens = 512;
    bool use_gpu = true;
    int n_threads = 4;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-m" || arg == "--model") {
            if (++i < argc) model_path = argv[i];
        } else if (arg == "-t" || arg == "--tokenizer") {
            if (++i < argc) tokenizer_path = argv[i];
        } else if (arg == "-i" || arg == "--image") {
            if (++i < argc) image_paths.push_back(argv[i]);
        } else if (arg == "--temperature") {
            if (++i < argc) temperature = std::stof(argv[i]);
        } else if (arg == "--max-tokens") {
            if (++i < argc) max_tokens = std::stoi(argv[i]);
        } else if (arg == "--cpu") {
            use_gpu = false;
        } else if (arg == "--threads") {
            if (++i < argc) n_threads = std::stoi(argv[i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (model_path.empty() || tokenizer_path.empty() || image_paths.empty()) {
        std::cerr << "Error: --model, --tokenizer, and --image are required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    try {
        // Initialize model
        std::cout << "Loading model..." << std::endl;
        latex_ocr::LatexOCR ocr(model_path, tokenizer_path);

        ocr.set_temperature(temperature);
        ocr.set_max_tokens(max_tokens);
        ocr.set_use_gpu(use_gpu);
        ocr.set_n_threads(n_threads);

        std::cout << std::endl;

        // Process each image
        for (const auto& image_path : image_paths) {
            std::cout << "--- " << image_path << " ---" << std::endl;

            std::string latex = ocr.recognize_file(image_path);

            if (latex.empty()) {
                std::cerr << "Failed to recognize: " << image_path << std::endl;
            } else {
                std::cout << "LaTeX: " << latex << std::endl;
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
