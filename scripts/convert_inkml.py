#!/usr/bin/env python3
"""
Convert CROHME InkML files to PNG images using OpenCV (fast rendering).

InkML files contain handwritten stroke data (x, y coordinate sequences).
This script renders them to grayscale PNG images suitable for LaTeX-OCR training.

Usage:
    python scripts/convert_inkml.py \
        --input /path/to/crohme_data \
        --output data/crohme_rendered \
        --line-width 3 \
        --padding 20 \
        --target-height 64
"""

import argparse
import os
import sys
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_inkml(filepath):
    """Parse InkML file and extract strokes + LaTeX ground truth.

    Args:
        filepath: Path to .inkml file

    Returns:
        tuple: (strokes, latex_gt)
        strokes: list of list of (x, y) tuples
        latex_gt: LaTeX string ground truth (or None)
    """
    try:
        tree = ET.parse(filepath)
    except ET.ParseError:
        return None, None

    root = tree.getroot()

    # Handle XML namespaces
    ns = {'ink': 'http://www.w3.org/2003/InkML'}

    # Try with namespace first, then without
    strokes = []
    traces = root.findall('.//ink:trace', ns)
    if not traces:
        traces = root.findall('.//{http://www.w3.org/2003/InkML}trace')
    if not traces:
        # Try without namespace
        traces = root.findall('.//trace')

    for trace in traces:
        text = trace.text
        if text is None:
            continue
        points = []
        for point_str in text.strip().split(','):
            parts = point_str.strip().split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append((x, y))
                except ValueError:
                    continue
        if points:
            strokes.append(points)

    # Extract LaTeX ground truth
    latex_gt = None
    # Try annotation elements
    for ann in root.findall('.//ink:annotation', ns) + root.findall('.//annotation'):
        ann_type = ann.get('type', '')
        if ann_type in ('truth', 'normalizedTruth', 'writer'):
            if ann_type != 'writer' and ann.text:
                latex_gt = ann.text.strip()
                # Remove $ delimiters if present
                if latex_gt.startswith('$') and latex_gt.endswith('$'):
                    latex_gt = latex_gt[1:-1].strip()
                break

    # Also try traceGroup/annotationXML for MathML (would need MathML->LaTeX converter)
    if latex_gt is None:
        for ann in root.findall('.//{http://www.w3.org/2003/InkML}annotation'):
            ann_type = ann.get('type', '')
            if ann_type == 'truth' and ann.text:
                latex_gt = ann.text.strip()
                if latex_gt.startswith('$') and latex_gt.endswith('$'):
                    latex_gt = latex_gt[1:-1].strip()
                break

    return strokes, latex_gt


def render_strokes_opencv(strokes, line_width=3, padding=20, target_height=None,
                           max_width=672, max_height=192):
    """Render handwritten strokes to a grayscale numpy array using OpenCV.

    This is ~100x faster than matplotlib for bulk rendering.

    Args:
        strokes: list of list of (x, y) tuples
        line_width: Width of rendered lines
        padding: Padding around the rendered strokes
        target_height: If set, resize to this height keeping aspect ratio
        max_width: Maximum allowed width
        max_height: Maximum allowed height

    Returns:
        numpy.ndarray: Grayscale image (H, W), uint8, white background with black strokes
    """
    if not strokes:
        return None

    # Compute bounding box
    all_x = [p[0] for s in strokes for p in s]
    all_y = [p[1] for s in strokes for p in s]

    if not all_x or not all_y:
        return None

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    width_range = max_x - min_x
    height_range = max_y - min_y

    if width_range < 1 or height_range < 1:
        return None

    # Determine canvas size
    if target_height:
        scale = (target_height - 2 * padding) / height_range
        canvas_w = int(width_range * scale) + 2 * padding
        canvas_h = target_height
    else:
        # Use natural scale but ensure reasonable size
        scale = 1.0
        # Scale to fit within max dimensions
        if width_range > max_width - 2 * padding or height_range > max_height - 2 * padding:
            scale_w = (max_width - 2 * padding) / width_range
            scale_h = (max_height - 2 * padding) / height_range
            scale = min(scale_w, scale_h)
        # Ensure minimum size
        if width_range * scale < 20 or height_range * scale < 20:
            scale = max(20 / width_range, 20 / height_range)

        canvas_w = int(width_range * scale) + 2 * padding
        canvas_h = int(height_range * scale) + 2 * padding

    # Clamp to max dimensions
    canvas_w = min(canvas_w, max_width)
    canvas_h = min(canvas_h, max_height)

    # Create white canvas
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

    # Render strokes
    for stroke in strokes:
        if len(stroke) < 2:
            continue

        # Convert to pixel coordinates
        pts = []
        for x, y in stroke:
            px = int((x - min_x) * scale + padding)
            py = int((y - min_y) * scale + padding)
            # Clamp to canvas
            px = max(0, min(px, canvas_w - 1))
            py = max(0, min(py, canvas_h - 1))
            pts.append((px, py))

        # Draw polyline
        pts_array = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_array], isClosed=False, color=0,
                      thickness=line_width, lineType=cv2.LINE_AA)

    return canvas


def process_crohme_directory(input_dir, output_dir, line_width=3, padding=20,
                              target_height=64, max_width=672, max_height=192):
    """Process all InkML files in a CROHME directory.

    Args:
        input_dir: Directory containing .inkml files
        output_dir: Output directory for rendered images

    Returns:
        list: List of (image_filename, latex) tuples
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Find all inkml files recursively
    inkml_files = []
    for ext in ['*.inkml', '*.InkML']:
        inkml_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

    print(f"Found {len(inkml_files)} InkML files in {input_dir}")

    samples = []
    skipped = 0

    for filepath in tqdm(inkml_files, desc="Rendering InkML"):
        strokes, latex_gt = parse_inkml(filepath)

        if strokes is None or latex_gt is None:
            skipped += 1
            continue

        if not latex_gt.strip():
            skipped += 1
            continue

        # Render strokes
        img = render_strokes_opencv(
            strokes,
            line_width=line_width,
            padding=padding,
            target_height=target_height,
            max_width=max_width,
            max_height=max_height
        )

        if img is None:
            skipped += 1
            continue

        # Save image
        idx = len(samples)
        img_filename = f"crohme_{idx:06d}.png"
        img_path = os.path.join(images_dir, img_filename)
        cv2.imwrite(img_path, img)

        samples.append((img_filename, latex_gt))

    # Save equations file
    eq_file = os.path.join(output_dir, "equations.txt")
    with open(eq_file, "w") as f:
        for img_name, latex in samples:
            f.write(f"{img_name}\t{latex}\n")

    print(f"Rendered {len(samples)} images, skipped {skipped}")
    print(f"Output: {images_dir}/")
    print(f"Equations: {eq_file}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Convert CROHME InkML files to PNG")
    parser.add_argument("--input", "-i", required=True,
                        help="Input directory containing .inkml files")
    parser.add_argument("--output", "-o", default="data/crohme_rendered",
                        help="Output directory for rendered images")
    parser.add_argument("--line-width", type=int, default=3,
                        help="Stroke line width (default: 3)")
    parser.add_argument("--padding", type=int, default=20,
                        help="Padding around strokes (default: 20)")
    parser.add_argument("--target-height", type=int, default=64,
                        help="Target image height (default: 64)")
    parser.add_argument("--max-width", type=int, default=672,
                        help="Maximum image width (default: 672)")
    parser.add_argument("--max-height", type=int, default=192,
                        help="Maximum image height (default: 192)")
    args = parser.parse_args()

    samples = process_crohme_directory(
        args.input, args.output,
        line_width=args.line_width,
        padding=args.padding,
        target_height=args.target_height,
        max_width=args.max_width,
        max_height=args.max_height
    )

    print(f"\nTotal: {len(samples)} handwritten expression images")


if __name__ == "__main__":
    main()
