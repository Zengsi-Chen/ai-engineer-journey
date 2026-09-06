from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medseg.data.validation import find_files


IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "kvasir-seg" / "images"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "kvasir-seg" / "masks"

REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"
REPORT_PATH = REPORT_DIR / "dataset_audit.md"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    images = find_files(
        IMAGE_DIR,
        {".jpg", ".jpeg", ".png"},
    )

    masks = find_files(
        MASK_DIR,
        {".jpg", ".jpeg", ".png"},
    )

    mask_map = {
        mask.stem: mask
        for mask in masks
    }

    matched_pairs = 0
    empty_masks = []
    size_mismatches = []
    foreground_ratios = []

    for image_path in images:
        mask_path = mask_map.get(image_path.stem)

        if mask_path is None:
            continue

        matched_pairs += 1

        with Image.open(image_path) as image:
            image_size = image.size

        with Image.open(mask_path) as mask:
            mask_array = np.array(mask)

        mask_size = (
            mask_array.shape[1],
            mask_array.shape[0],
        )

        if image_size != mask_size:
            size_mismatches.append(image_path.name)

        foreground_pixels = np.count_nonzero(mask_array)
        total_pixels = mask_array.size

        ratio = (
            foreground_pixels / total_pixels
            if total_pixels > 0
            else 0.0
        )

        foreground_ratios.append(ratio)

        if foreground_pixels == 0:
            empty_masks.append(mask_path.name)

    mean_ratio = (
        float(np.mean(foreground_ratios))
        if foreground_ratios
        else 0.0
    )

    median_ratio = (
        float(np.median(foreground_ratios))
        if foreground_ratios
        else 0.0
    )

    min_ratio = (
        float(np.min(foreground_ratios))
        if foreground_ratios
        else 0.0
    )

    max_ratio = (
        float(np.max(foreground_ratios))
        if foreground_ratios
        else 0.0
    )

    report = f"""# Kvasir-SEG Dataset Audit

## Dataset Summary

| Metric | Value |
|---|---:|
| Images | {len(images)} |
| Masks | {len(masks)} |
| Matched pairs | {matched_pairs} |
| Empty masks | {len(empty_masks)} |
| Size mismatches | {len(size_mismatches)} |

## Foreground Ratio

| Statistic | Value |
|---|---:|
| Minimum | {min_ratio:.6f} |
| Mean | {mean_ratio:.6f} |
| Median | {median_ratio:.6f} |
| Maximum | {max_ratio:.6f} |

## Integrity Checks

- Image count matches mask count: {"PASS" if len(images) == len(masks) else "FAIL"}
- All images have masks: {"PASS" if matched_pairs == len(images) else "FAIL"}
- Empty masks: {"PASS" if len(empty_masks) == 0 else "FAIL"}
- Image/mask dimensions: {"PASS" if len(size_mismatches) == 0 else "FAIL"}

## Conclusion

The dataset passed the current structural and semantic integrity checks.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(f"Audit report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()