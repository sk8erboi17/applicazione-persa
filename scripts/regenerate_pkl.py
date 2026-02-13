#!/usr/bin/env python3
"""Regenerate train.pkl and val.pkl from existing data/unified/ without reprocessing datasets.
Use this after fixing pickle-incompatible code (e.g., albumentations version changes)."""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pix2tex.dataset.dataset import Im2LatexDataset


def main():
    parser = argparse.ArgumentParser(description="Regenerate pkl files from existing unified dataset")
    parser.add_argument("--data-dir", type=str, default="data/unified")
    parser.add_argument("--max-width", type=int, default=672)
    parser.add_argument("--max-height", type=int, default=192)
    args = parser.parse_args()

    data_dir = os.path.join(str(PROJECT_ROOT), args.data_dir)
    tokenizer_path = os.path.join(data_dir, "tokenizer.json")

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: tokenizer not found at {tokenizer_path}")
        sys.exit(1)

    for split in ["train", "val"]:
        eq_file = os.path.join(data_dir, split, "equations.txt")
        img_dir = os.path.join(data_dir, split, "images")
        pkl_file = os.path.join(data_dir, f"{split}.pkl")

        if not os.path.exists(eq_file) or not os.path.isdir(img_dir):
            print(f"SKIP: {split} — missing equations.txt or images/")
            continue

        print(f"--- Regenerating {split}.pkl ---")
        dataset = Im2LatexDataset(
            equations=eq_file, images=img_dir, tokenizer=tokenizer_path,
            shuffle=True, batchsize=1, max_seq_len=512,
            max_dimensions=(args.max_width, args.max_height),
            min_dimensions=(32, 32), keep_smaller_batches=True, test=False,
        )
        dataset.save(pkl_file)
        print(f"  {split}.pkl: {len(dataset)} entries saved")

    print("\nDone! You can now run training.")


if __name__ == "__main__":
    main()
