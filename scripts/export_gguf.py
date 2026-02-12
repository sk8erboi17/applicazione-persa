#!/usr/bin/env python3
"""
Export a trained LaTeX-OCR PyTorch model to GGUF format for C++ inference with ggml.

Usage:
    python scripts/export_gguf.py \
        --checkpoint checkpoints/latex_ocr_large/best.pth \
        --config pix2tex/model/settings/config_large.yaml \
        --output model.gguf \
        --dtype fp16
"""

import argparse
import os
import sys
import struct
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from munch import Munch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pix2tex.models import get_model
from pix2tex.utils import parse_args

# GGUF constants
GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8


def write_string(f, s):
    """Write a GGUF string (length-prefixed)."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_kv_string(f, key, value):
    """Write a key-value pair with string value."""
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_STRING))
    write_string(f, value)


def write_kv_uint32(f, key, value):
    """Write a key-value pair with uint32 value."""
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_UINT32))
    f.write(struct.pack('<I', value))


def write_kv_int32(f, key, value):
    """Write a key-value pair with int32 value."""
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_INT32))
    f.write(struct.pack('<i', value))


def write_kv_float32(f, key, value):
    """Write a key-value pair with float32 value."""
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_FLOAT32))
    f.write(struct.pack('<f', value))


def write_kv_bool(f, key, value):
    """Write a key-value pair with bool value."""
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_BOOL))
    f.write(struct.pack('<?', value))


def write_kv_array_int32(f, key, values):
    """Write a key-value pair with array of int32 values."""
    write_string(f, key)
    f.write(struct.pack('<I', GGUF_TYPE_ARRAY))
    f.write(struct.pack('<I', GGUF_TYPE_INT32))
    f.write(struct.pack('<Q', len(values)))
    for v in values:
        f.write(struct.pack('<i', v))


def quantize_q8_0(tensor_data, block_size=32):
    """Quantize float32 tensor to Q8_0 format.

    Q8_0: each block of 32 values is stored as:
    - 1 float16 scale factor (2 bytes)
    - 32 int8 quantized values (32 bytes)
    Total: 34 bytes per 32 values
    """
    flat = tensor_data.flatten().astype(np.float32)
    # Pad to multiple of block_size
    remainder = len(flat) % block_size
    if remainder != 0:
        flat = np.concatenate([flat, np.zeros(block_size - remainder, dtype=np.float32)])

    n_blocks = len(flat) // block_size
    blocks = flat.reshape(n_blocks, block_size)

    output = bytearray()
    for block in blocks:
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax != 0 else 0.0
        # Write scale as float16
        output += np.float16(scale).tobytes()
        # Quantize and write int8 values
        if scale != 0:
            quantized = np.round(block / scale).clip(-128, 127).astype(np.int8)
        else:
            quantized = np.zeros(block_size, dtype=np.int8)
        output += quantized.tobytes()

    return bytes(output)


def export_gguf(checkpoint_path, config_path, output_path, dtype='fp16'):
    """Export PyTorch model to GGUF format."""

    # Load config
    with open(config_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params), no_cuda=True)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = get_model(args)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # Collect all tensors
    tensors = {}
    for name, param in model.named_parameters():
        tensors[name] = param.detach().cpu().numpy()
    for name, buf in model.named_buffers():
        tensors[name] = buf.detach().cpu().numpy()

    print(f"Model has {len(tensors)} tensors")
    total_params = sum(t.size for t in tensors.values())
    print(f"Total parameters: {total_params:,}")

    # Determine tensor type
    if dtype == 'fp32':
        ggml_type = GGML_TYPE_F32
        type_size = 4
    elif dtype == 'fp16':
        ggml_type = GGML_TYPE_F16
        type_size = 2
    elif dtype == 'q8_0':
        ggml_type = GGML_TYPE_Q8_0
        type_size = 0  # Variable, handled separately
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Prepare metadata key-value pairs
    metadata = [
        ('general.architecture', 'latex_ocr', 'string'),
        ('general.name', 'LaTeX-OCR', 'string'),
        ('general.file_type', {'fp32': 0, 'fp16': 1, 'q8_0': 8}[dtype], 'uint32'),
        ('latex_ocr.context_length', args.max_seq_len, 'uint32'),
        ('latex_ocr.embedding_length', args.dim, 'uint32'),
        ('latex_ocr.encoder.depth', args.encoder_depth, 'uint32'),
        ('latex_ocr.decoder.depth', args.num_layers, 'uint32'),
        ('latex_ocr.attention.head_count', args.heads, 'uint32'),
        ('latex_ocr.vocab_size', args.num_tokens, 'uint32'),
        ('latex_ocr.patch_size', args.patch_size, 'uint32'),
        ('latex_ocr.image.max_width', args.max_width, 'uint32'),
        ('latex_ocr.image.max_height', args.max_height, 'uint32'),
        ('latex_ocr.image.channels', args.channels, 'uint32'),
        ('latex_ocr.backbone_layers', list(args.backbone_layers), 'array_int32'),
        ('latex_ocr.encoder_structure', args.encoder_structure, 'string'),
        ('latex_ocr.decoder.attn_on_attn', args.decoder_args.get('attn_on_attn', False), 'bool'),
        ('latex_ocr.decoder.ff_glu', args.decoder_args.get('ff_glu', False), 'bool'),
        ('latex_ocr.decoder.cross_attend', args.decoder_args.get('cross_attend', True), 'bool'),
        ('latex_ocr.token.pad', args.pad_token, 'int32'),
        ('latex_ocr.token.bos', args.bos_token, 'int32'),
        ('latex_ocr.token.eos', args.eos_token, 'int32'),
        ('latex_ocr.temperature', args.temperature, 'float32'),
    ]

    n_kv = len(metadata)
    n_tensors = len(tensors)

    print(f"Writing GGUF to {output_path}...")
    print(f"  Format: {dtype}")
    print(f"  Metadata entries: {n_kv}")
    print(f"  Tensor count: {n_tensors}")

    with open(output_path, 'wb') as f:
        # === HEADER ===
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', n_tensors))
        f.write(struct.pack('<Q', n_kv))

        # === METADATA KEY-VALUE PAIRS ===
        for key, value, vtype in metadata:
            if vtype == 'string':
                write_kv_string(f, key, value)
            elif vtype == 'uint32':
                write_kv_uint32(f, key, value)
            elif vtype == 'int32':
                write_kv_int32(f, key, value)
            elif vtype == 'float32':
                write_kv_float32(f, key, value)
            elif vtype == 'bool':
                write_kv_bool(f, key, value)
            elif vtype == 'array_int32':
                write_kv_array_int32(f, key, value)

        # === TENSOR INFO ===
        # First pass: write tensor metadata
        tensor_data_list = []
        offset = 0

        for name, data in tensors.items():
            write_string(f, name)  # tensor name
            ndim = len(data.shape)
            f.write(struct.pack('<I', ndim))  # n_dimensions

            # Write dimensions (GGUF uses reverse order from numpy)
            for dim in reversed(data.shape):
                f.write(struct.pack('<Q', dim))

            # Determine type for this tensor
            # Use FP32 for 1D tensors (biases, norms) even when quantizing
            if dtype == 'q8_0' and ndim >= 2:
                tensor_type = GGML_TYPE_Q8_0
                q_data = quantize_q8_0(data)
                tensor_data_list.append(q_data)
                data_size = len(q_data)
            elif dtype == 'fp16':
                tensor_type = GGML_TYPE_F16
                fp16_data = data.astype(np.float16).tobytes()
                tensor_data_list.append(fp16_data)
                data_size = len(fp16_data)
            else:
                tensor_type = GGML_TYPE_F32
                fp32_data = data.astype(np.float32).tobytes()
                tensor_data_list.append(fp32_data)
                data_size = len(fp32_data)

            f.write(struct.pack('<I', tensor_type))  # type

            # Align offset to 32 bytes
            offset = (offset + 31) & ~31
            f.write(struct.pack('<Q', offset))  # offset from start of data section

            offset += data_size

        # === TENSOR DATA ===
        # Align to 32 bytes before data section
        current_pos = f.tell()
        alignment = 32
        padding = (alignment - (current_pos % alignment)) % alignment
        f.write(b'\x00' * padding)

        data_start = f.tell()

        for i, (name, _) in enumerate(tensors.items()):
            # Align each tensor to 32 bytes
            current_pos = f.tell() - data_start
            padding = (32 - (current_pos % 32)) % 32
            f.write(b'\x00' * padding)

            f.write(tensor_data_list[i])

    file_size = os.path.getsize(output_path)
    print(f"\nExport complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size / (1024*1024):.1f} MB")

    if dtype == 'fp32':
        expected = total_params * 4
    elif dtype == 'fp16':
        expected = total_params * 2
    else:
        expected = total_params  # rough estimate for q8_0

    print(f"  Compression: {file_size/expected:.2f}x vs raw {dtype}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export LaTeX-OCR model to GGUF')
    parser.add_argument('--checkpoint', '-c', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', default='pix2tex/model/settings/config_large.yaml',
                        help='Path to config YAML')
    parser.add_argument('--output', '-o', default='model.gguf', help='Output GGUF file path')
    parser.add_argument('--dtype', choices=['fp32', 'fp16', 'q8_0'], default='fp16',
                        help='Data type for export (default: fp16)')
    args = parser.parse_args()
    export_gguf(args.checkpoint, args.config, args.output, args.dtype)
