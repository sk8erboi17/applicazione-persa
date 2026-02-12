#!/bin/bash
# ════════════════════════════════════════════════════════════
# LaTeX-OCR C++ Inference Engine - Setup Script
# ════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== LaTeX-OCR C++ Setup ==="
echo ""

# 1. Clone ggml if not present
if [ ! -d "third_party/ggml" ]; then
    echo "[1/3] Cloning ggml..."
    git clone --depth 1 https://github.com/ggml-org/ggml.git third_party/ggml
else
    echo "[1/3] ggml already present"
fi

# 2. Check third-party deps
echo "[2/3] Checking dependencies..."
for f in third_party/stb_image.h third_party/cJSON.h third_party/cJSON.c; do
    if [ ! -f "$f" ]; then
        echo "  MISSING: $f"
        exit 1
    fi
done
echo "  All dependencies OK"

# 3. Build
echo "[3/3] Building..."
mkdir -p build
cd build

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  Detected macOS - enabling Metal + Accelerate"
    cmake .. -DLATEX_OCR_METAL=ON -DLATEX_OCR_ACCELERATE=ON -DCMAKE_BUILD_TYPE=Release
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "  Detected Linux - disabling Metal, using OpenBLAS"
    cmake .. -DLATEX_OCR_METAL=OFF -DLATEX_OCR_ACCELERATE=OFF -DCMAKE_BUILD_TYPE=Release
else
    echo "  Unknown OS: $OSTYPE - building with defaults"
    cmake .. -DCMAKE_BUILD_TYPE=Release
fi

make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo ""
echo "=== Build complete! ==="
echo "Binary: $(pwd)/latex_ocr"
echo ""
echo "Usage:"
echo "  ./latex_ocr -m model.gguf -t tokenizer.json -i image.png"
