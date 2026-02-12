#!/usr/bin/env python3
"""
Prepare a unified dataset for LaTeX-OCR training from multiple sources.
Memory-efficient: saves images to disk immediately, keeps only (formula, tag) in RAM.

Datasets supported:
- OleehyO/latex-formulas (cleaned_formulas): 552k samples [HuggingFace]
- wanderkid/UniMER_Dataset: 1.06M samples [HuggingFace]
- yuntian-deng/im2latex-100k: 68k processed samples [HuggingFace]
- lukbl/LaTeX-OCR-dataset: 158k samples [HuggingFace]
- deepcopy/MathWriting-human: 253k handwritten samples [HuggingFace]
- hoang-quoc-trung/fusion-image-to-latex-datasets: 3.4M train [local CSV + images]
- HME100K: ~99k handwritten samples [local zip]
"""

import argparse
import csv
import json
import os
import sys
import pickle
import zipfile
import io
from pathlib import Path

import cv2
import imagesize
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pix2tex.dataset.dataset import Im2LatexDataset, generate_tokenizer


# ──────────────────────────────────────────────────────────────────
# Streaming loaders: save images to disk, return (formula, tag) only
# ──────────────────────────────────────────────────────────────────

def _save_pil_image(img, idx, images_dir, max_w=672, max_h=192, min_w=32, min_h=32):
    """Save a PIL image to disk. Returns True if saved, False if skipped."""
    try:
        img_np = np.array(img.convert("RGB"))
    except Exception:
        return False

    h, w = img_np.shape[:2]
    if w < min_w or w > max_w or h < min_h or h > max_h:
        return False

    img_path = os.path.join(images_dir, f"{idx:07d}.png")
    if len(img_np.shape) == 3:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_np
    cv2.imwrite(img_path, img_bgr)
    return True


def stream_oleeh(images_dir, idx_start, cache_dir, max_w, max_h):
    """Stream OleehyO dataset: save images to disk, yield (formula, tag, saved)."""
    from datasets import load_dataset
    print("[HF] Loading OleehyO/latex-formulas (cleaned_formulas)...")
    ds = load_dataset("OleehyO/latex-formulas", "cleaned_formulas", cache_dir=cache_dir)
    saved, skipped = 0, 0
    idx = idx_start
    formulas_tags = []
    for split in ds:
        for item in tqdm(ds[split], desc=f"  OleehyO/{split}"):
            try:
                img = item.get("image")
                formula = item.get("latex_formula", "")
                if img is not None and formula.strip():
                    if _save_pil_image(img, idx, images_dir, max_w, max_h):
                        formulas_tags.append((formula.strip(), "printed"))
                        idx += 1
                        saved += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1
    print(f"  -> {saved} saved, {skipped} skipped")
    return formulas_tags, idx


def stream_unimer(images_dir, idx_start, cache_dir, max_w, max_h):
    """Stream UniMER: try ZIP first, then fallback to HuggingFace load_dataset."""
    print("[HF] Loading wanderkid/UniMER_Dataset...")

    # Try ZIP method first (faster, works on Python 3.14)
    hf_cache = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    ds_dir = os.path.join(hf_cache, "datasets--wanderkid--UniMER_Dataset")

    zip_path = None
    if os.path.isdir(ds_dir):
        for root, dirs, files in os.walk(os.path.join(ds_dir, "snapshots")):
            for f in files:
                if f == "UniMER-1M.zip":
                    zip_path = os.path.realpath(os.path.join(root, f))
                    break

    if zip_path and os.path.exists(zip_path):
        print(f"  Found ZIP: {zip_path}")
        saved, skipped = 0, 0
        idx = idx_start
        formulas_tags = []

        try:
            zf = zipfile.ZipFile(zip_path, "r")
            with zf.open("UniMER-1M/train.txt") as f:
                formulas = [line.decode("utf-8").strip() for line in f.readlines()]
            available = set(n for n in zf.namelist() if n.endswith(".png"))
            print(f"  {len(formulas)} formulas, {len(available)} images")

            for i, formula in enumerate(tqdm(formulas, desc="  UniMER")):
                if not formula:
                    continue
                img_name = f"UniMER-1M/images/{i:07d}.png"
                if img_name not in available:
                    skipped += 1
                    continue
                try:
                    img_data = zf.read(img_name)
                    img = Image.open(io.BytesIO(img_data))
                    img.load()
                    if _save_pil_image(img, idx, images_dir, max_w, max_h):
                        formulas_tags.append((formula, "printed"))
                        idx += 1
                        saved += 1
                    else:
                        skipped += 1
                    del img_data, img
                except Exception:
                    skipped += 1
            zf.close()
        except Exception as e:
            print(f"  ZIP error: {e}")

        print(f"  -> {saved} saved, {skipped} skipped")
        return formulas_tags, idx

    # Fallback: download via HuggingFace load_dataset
    print("  ZIP not found, downloading via HuggingFace...")
    from datasets import load_dataset
    try:
        ds = load_dataset("wanderkid/UniMER_Dataset", "UniMER-1M", cache_dir=cache_dir)
    except Exception:
        try:
            ds = load_dataset("wanderkid/UniMER_Dataset", cache_dir=cache_dir)
        except Exception as e:
            print(f"  Failed to load UniMER via HF: {e}")
            return [], idx_start

    saved, skipped = 0, 0
    idx = idx_start
    formulas_tags = []

    for split in ds:
        n = len(ds[split])
        for i in tqdm(range(n), desc=f"  UniMER/{split}"):
            try:
                item = ds[split][i]
                img = item.get("image")
                formula = item.get("latex", item.get("formula", item.get("text", "")))
                if img is not None and formula and formula.strip():
                    if _save_pil_image(img, idx, images_dir, max_w, max_h):
                        formulas_tags.append((formula.strip(), "printed"))
                        idx += 1
                        saved += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    print(f"  -> {saved} saved, {skipped} skipped")
    return formulas_tags, idx


def stream_im2latex(images_dir, idx_start, cache_dir, max_w, max_h):
    """Stream im2latex-100k: save images to disk."""
    from datasets import load_dataset
    print("[HF] Loading yuntian-deng/im2latex-100k...")
    ds = load_dataset("yuntian-deng/im2latex-100k", cache_dir=cache_dir)
    saved, skipped = 0, 0
    idx = idx_start
    formulas_tags = []
    for split in ds:
        for item in tqdm(ds[split], desc=f"  im2latex/{split}"):
            try:
                img = item.get("image")
                formula = item.get("formula", item.get("text", ""))
                if img is not None and formula.strip():
                    if _save_pil_image(img, idx, images_dir, max_w, max_h):
                        formulas_tags.append((formula.strip(), "printed"))
                        idx += 1
                        saved += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1
    print(f"  -> {saved} saved, {skipped} skipped")
    return formulas_tags, idx


def stream_lukbl(images_dir, idx_start, cache_dir, max_w, max_h):
    """Stream lukbl dataset by index to handle corrupt images."""
    from datasets import load_dataset
    print("[HF] Loading lukbl/LaTeX-OCR-dataset...")
    ds = load_dataset("lukbl/LaTeX-OCR-dataset", cache_dir=cache_dir)
    saved, skipped = 0, 0
    idx = idx_start
    formulas_tags = []
    for split in ds:
        n = len(ds[split])
        for i in tqdm(range(n), desc=f"  lukbl/{split}"):
            try:
                item = ds[split][i]
                img = item.get("image")
                formula = item.get("text", item.get("formula", ""))
                if img is not None and formula.strip():
                    img.load()
                    if _save_pil_image(img, idx, images_dir, max_w, max_h):
                        formulas_tags.append((formula.strip(), "printed"))
                        idx += 1
                        saved += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1
    print(f"  -> {saved} saved, {skipped} skipped")
    return formulas_tags, idx


def stream_fusion(images_dir, idx_start, fusion_base_dir, images_root_dir, max_w, max_h):
    """Stream fusion dataset from local CSV + images."""
    print("[LOCAL] Loading fusion-image-to-latex-datasets...")
    saved, skipped = 0, 0
    idx = idx_start
    formulas_tags = []

    for split in ["train", "val"]:
        csv_dir = os.path.join(fusion_base_dir, split)
        if not os.path.isdir(csv_dir):
            continue
        for csv_file in sorted(os.listdir(csv_dir)):
            if not csv_file.endswith(".csv"):
                continue
            csv_path = os.path.join(csv_dir, csv_file)
            is_hw = "handwritten" in csv_file
            with open(csv_path, "r") as f:
                total = sum(1 for _ in f) - 1
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in tqdm(reader, desc=f"  fusion/{split}/{csv_file}", total=total):
                    img_filename = row.get("image_filename", "")
                    latex = row.get("latex", "")
                    if not img_filename or not latex.strip():
                        continue
                    img_path = os.path.join(images_root_dir, img_filename)
                    if not os.path.exists(img_path):
                        skipped += 1
                        continue
                    try:
                        img = Image.open(img_path)
                        img.load()
                        tag = "handwritten" if is_hw else "printed"
                        if _save_pil_image(img, idx, images_dir, max_w, max_h):
                            formulas_tags.append((latex.strip(), tag))
                            idx += 1
                            saved += 1
                        else:
                            skipped += 1
                        del img
                    except Exception:
                        skipped += 1

    print(f"  -> {saved} saved, {skipped} skipped")
    return formulas_tags, idx


def stream_hme100k(images_dir, idx_start, zip_path, extract_dir, max_w, max_h):
    """Stream HME100K from extracted dir."""
    print(f"[LOCAL] Loading HME100K from {zip_path}...")
    if extract_dir is None:
        extract_dir = os.path.join(os.path.dirname(zip_path), "hme100k_extracted")

    if not os.path.isdir(os.path.join(extract_dir, "images")):
        print(f"  Extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    train_txt = os.path.join(extract_dir, "train.txt")
    if not os.path.exists(train_txt):
        print(f"  Error: {train_txt} not found")
        return [], idx_start

    saved, skipped = 0, 0
    idx = idx_start
    formulas_tags = []

    with open(train_txt, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="  HME100K"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        img_rel_path, latex = parts
        img_path = os.path.join(extract_dir, img_rel_path)
        if not os.path.exists(img_path) or not latex.strip():
            skipped += 1
            continue
        try:
            img = Image.open(img_path)
            img.load()
            if _save_pil_image(img, idx, images_dir, max_w, max_h):
                formulas_tags.append((latex.strip(), "handwritten"))
                idx += 1
                saved += 1
            else:
                skipped += 1
            del img
        except Exception:
            skipped += 1

    print(f"  -> {saved} saved, {skipped} skipped")
    return formulas_tags, idx


def stream_mathwriting(images_dir, idx_start, cache_dir, max_w, max_h):
    """Stream MathWriting-human dataset from HuggingFace (~253k handwritten)."""
    from datasets import load_dataset
    print("[HF] Loading deepcopy/MathWriting-human...")
    try:
        ds = load_dataset("deepcopy/MathWriting-human", cache_dir=cache_dir)
    except Exception as e:
        print(f"  Failed to load MathWriting: {e}")
        return [], idx_start

    saved, skipped = 0, 0
    idx = idx_start
    formulas_tags = []

    for split in ds:
        n = len(ds[split])
        for i in tqdm(range(n), desc=f"  MathWriting/{split}"):
            try:
                item = ds[split][i]
                img = item.get("image")
                formula = item.get("latex", "")
                if img is not None and formula and formula.strip():
                    if _save_pil_image(img, idx, images_dir, max_w, max_h):
                        formulas_tags.append((formula.strip(), "handwritten"))
                        idx += 1
                        saved += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    print(f"  -> {saved} saved, {skipped} skipped")
    return formulas_tags, idx


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare unified dataset for LaTeX-OCR training")
    parser.add_argument("--output", type=str, default="data/unified")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-width", type=int, default=672)
    parser.add_argument("--max-height", type=int, default=192)
    parser.add_argument("--fusion-dir", type=str, default=None)
    parser.add_argument("--fusion-images", type=str, default=None)
    parser.add_argument("--hme100k-zip", type=str, default=None)
    parser.add_argument("--hme100k-extract", type=str, default=None)
    parser.add_argument("--skip-datasets", type=str, nargs="*", default=[])
    parser.add_argument("--hw-ratio", type=float, default=0.2)
    args = parser.parse_args()

    output_dir = os.path.join(str(PROJECT_ROOT), args.output)
    os.makedirs(output_dir, exist_ok=True)

    # ── Phase 1: Stream all datasets to a staging directory ──
    # Save images immediately to disk, keep only (formula, tag) in RAM
    staging_dir = os.path.join(output_dir, "staging_images")
    os.makedirs(staging_dir, exist_ok=True)

    all_formulas_tags = []  # list of (formula, tag) — lightweight strings only
    idx = 0

    if "oleeh" not in args.skip_datasets:
        ft, idx = stream_oleeh(staging_dir, idx, args.cache_dir, args.max_width, args.max_height)
        all_formulas_tags.extend(ft)
        del ft

    if "unimer" not in args.skip_datasets:
        ft, idx = stream_unimer(staging_dir, idx, args.cache_dir, args.max_width, args.max_height)
        all_formulas_tags.extend(ft)
        del ft

    if "im2latex" not in args.skip_datasets:
        ft, idx = stream_im2latex(staging_dir, idx, args.cache_dir, args.max_width, args.max_height)
        all_formulas_tags.extend(ft)
        del ft

    if "lukbl" not in args.skip_datasets:
        ft, idx = stream_lukbl(staging_dir, idx, args.cache_dir, args.max_width, args.max_height)
        all_formulas_tags.extend(ft)
        del ft

    if "fusion" not in args.skip_datasets and args.fusion_dir and args.fusion_images:
        ft, idx = stream_fusion(staging_dir, idx, args.fusion_dir, args.fusion_images, args.max_width, args.max_height)
        all_formulas_tags.extend(ft)
        del ft

    if "hme100k" not in args.skip_datasets and args.hme100k_zip:
        ft, idx = stream_hme100k(staging_dir, idx, args.hme100k_zip, args.hme100k_extract, args.max_width, args.max_height)
        all_formulas_tags.extend(ft)
        del ft

    if "mathwriting" not in args.skip_datasets:
        ft, idx = stream_mathwriting(staging_dir, idx, args.cache_dir, args.max_width, args.max_height)
        all_formulas_tags.extend(ft)
        del ft

    n_total = len(all_formulas_tags)
    n_hw = sum(1 for _, t in all_formulas_tags if t == "handwritten")
    n_pr = n_total - n_hw
    print(f"\nTotal saved to staging: {n_total}")
    print(f"  Printed:     {n_pr}")
    print(f"  Handwritten: {n_hw}")

    # ── Phase 2: Deduplicate by formula ──
    print("Deduplicating by formula...")
    seen = set()
    unique_indices = []
    for i, (formula, tag) in enumerate(all_formulas_tags):
        if formula not in seen:
            seen.add(formula)
            unique_indices.append(i)
    del seen
    print(f"  After dedup: {len(unique_indices)} unique samples (removed {n_total - len(unique_indices)} duplicates)")

    # ── Phase 3: Shuffle and split ──
    np.random.seed(42)
    np.random.shuffle(unique_indices)

    val_size = int(len(unique_indices) * args.val_split)
    val_indices = unique_indices[:val_size]
    train_indices = unique_indices[val_size:]
    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")

    # ── Phase 4: Move images from staging to train/val ──
    def move_split(split_indices, split_name):
        split_dir = os.path.join(output_dir, split_name)
        split_images = os.path.join(split_dir, "images")
        os.makedirs(split_images, exist_ok=True)

        equations = []
        tags = []
        new_idx = 0

        for old_idx in tqdm(split_indices, desc=f"Organizing {split_name}"):
            formula, tag = all_formulas_tags[old_idx]
            old_path = os.path.join(staging_dir, f"{old_idx:07d}.png")
            new_path = os.path.join(split_images, f"{new_idx:07d}.png")

            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                equations.append(formula)
                tags.append(tag)
                new_idx += 1

        # Save equations
        eq_file = os.path.join(split_dir, "equations.txt")
        with open(eq_file, "w") as f:
            f.write("\n".join(equations))

        # Save tags
        tags_file = os.path.join(split_dir, "tags.json")
        with open(tags_file, "w") as f:
            json.dump(tags, f)

        n_hw_s = sum(1 for t in tags if t == "handwritten")
        n_pr_s = sum(1 for t in tags if t == "printed")
        print(f"  {split_name}: {new_idx} samples ({n_pr_s} printed, {n_hw_s} handwritten)")
        return eq_file, split_images, new_idx, tags

    print("\n--- Organizing train/val splits ---")
    train_eq, train_imgs, train_count, train_tags = move_split(train_indices, "train")
    val_eq, val_imgs, val_count, val_tags = move_split(val_indices, "val")

    # Clean up staging (remaining duplicate images)
    import shutil
    shutil.rmtree(staging_dir, ignore_errors=True)

    # ── Phase 5: Compute sampling weights ──
    print("\n--- Computing sampling weights ---")
    n_train_hw = sum(1 for t in train_tags if t == "handwritten")
    n_train_pr = sum(1 for t in train_tags if t == "printed")
    target_hw_ratio = args.hw_ratio

    if n_train_hw > 0 and n_train_pr > 0:
        w_hw = target_hw_ratio * n_train_pr / ((1 - target_hw_ratio) * n_train_hw)
        w_pr = 1.0
    else:
        w_hw, w_pr = 1.0, 1.0

    sample_weights = [w_hw if t == "handwritten" else w_pr for t in train_tags]

    weights_file = os.path.join(output_dir, "sample_weights.json")
    with open(weights_file, "w") as f:
        json.dump({
            "weights": sample_weights,
            "n_printed": n_train_pr,
            "n_handwritten": n_train_hw,
            "w_printed": w_pr,
            "w_handwritten": w_hw,
            "target_hw_ratio": target_hw_ratio,
        }, f)
    print(f"  Printed weight:     {w_pr:.4f} (n={n_train_pr})")
    print(f"  Handwritten weight: {w_hw:.4f} (n={n_train_hw})")
    print(f"  Target HW ratio per batch: {target_hw_ratio:.1%}")

    # ── Phase 6: Train tokenizer ──
    print("\n--- Training BPE tokenizer ---")
    all_equations_file = os.path.join(output_dir, "all_equations.txt")
    with open(all_equations_file, "w") as f:
        for i in unique_indices:
            formula, _ = all_formulas_tags[i]
            f.write(formula + "\n")

    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    generate_tokenizer([all_equations_file], tokenizer_path, args.vocab_size)
    print(f"  Tokenizer saved: {tokenizer_path}")

    # ── Phase 7: Create pickle datasets ──
    print("\n--- Creating train.pkl ---")
    dataset = Im2LatexDataset(
        equations=train_eq, images=train_imgs, tokenizer=tokenizer_path,
        shuffle=True, batchsize=1, max_seq_len=512,
        max_dimensions=(args.max_width, args.max_height),
        min_dimensions=(32, 32), keep_smaller_batches=True, test=False,
    )
    dataset.save(os.path.join(output_dir, "train.pkl"))
    print(f"  Train dataset: {len(dataset)} batches")

    print("\n--- Creating val.pkl ---")
    dataset = Im2LatexDataset(
        equations=val_eq, images=val_imgs, tokenizer=tokenizer_path,
        shuffle=True, batchsize=1, max_seq_len=512,
        max_dimensions=(args.max_width, args.max_height),
        min_dimensions=(32, 32), keep_smaller_batches=True, test=False,
    )
    dataset.save(os.path.join(output_dir, "val.pkl"))
    print(f"  Val dataset: {len(dataset)} batches")

    # ── Phase 8: Save metadata ──
    metadata = {
        "total_train": train_count,
        "total_val": val_count,
        "train_printed": n_train_pr,
        "train_handwritten": n_train_hw,
        "vocab_size": args.vocab_size,
        "max_width": args.max_width,
        "max_height": args.max_height,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Summary ──
    total_unique = len(unique_indices)
    total_saved = train_count + val_count
    total_skipped_dedup = n_total - total_unique

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Total loaded:            {n_total}")
    print(f"  Removed (duplicates):    {total_skipped_dedup}")
    print(f"  Unique:                  {total_unique}")
    print(f"  Final saved:             {total_saved}")
    print(f"  Train samples:  {train_count} ({n_train_pr} printed + {n_train_hw} handwritten)")
    print(f"  Val samples:    {val_count}")
    print(f"  Tokenizer:      {tokenizer_path}")
    print(f"  Sample weights: {weights_file}")
    print(f"  Train pkl:      {os.path.join(output_dir, 'train.pkl')}")
    print(f"  Val pkl:        {os.path.join(output_dir, 'val.pkl')}")
    print()


if __name__ == "__main__":
    main()
