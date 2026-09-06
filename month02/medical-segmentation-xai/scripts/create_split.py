from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from medseg.data.split import (
    build_image_mask_pairs,
    find_duplicate_images,
    split_pairs,
)


IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kvasir-seg"
    / "images"
)

MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kvasir-seg"
    / "masks"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
)


def save_csv(
    rows: list[dict],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "sample_id",
        "image_path",
        "mask_path",
        "split",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


def main():

    print("Building image-mask pairs...")

    pairs = build_image_mask_pairs(
        IMAGE_DIR,
        MASK_DIR,
    )

    print(f"Total pairs: {len(pairs)}")

    duplicates = find_duplicate_images(
        pairs
    )

    if duplicates:

        print("ERROR: Duplicate images detected:")

        for file_hash, sample_ids in duplicates.items():
            print(
                f"Hash: {file_hash}"
            )
            print(
                f"Samples: {sample_ids}"
            )

        raise RuntimeError(
            "Duplicate images detected. "
            "Split generation aborted."
        )

    print("Duplicate check: PASS")

    splits = split_pairs(
        pairs,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    print(
        f"Train: {len(splits['train'])}"
    )

    print(
        f"Validation: {len(splits['val'])}"
    )

    print(
        f"Test: {len(splits['test'])}"
    )

    manifest = []

    for split_name in [
        "train",
        "val",
        "test",
    ]:

        for pair in splits[split_name]:

            row = {
                **pair,
                "split": split_name,
            }

            manifest.append(row)

    train_rows = [
        {**item, "split": "train"}
        for item in splits["train"]
    ]

    val_rows = [
        {**item, "split": "val"}
        for item in splits["val"]
    ]

    test_rows = [
        {**item, "split": "test"}
        for item in splits["test"]
    ]

    save_csv(
        train_rows,
        OUTPUT_DIR / "train.csv",
    )

    save_csv(
        val_rows,
        OUTPUT_DIR / "val.csv",
    )

    save_csv(
        test_rows,
        OUTPUT_DIR / "test.csv",
    )

    save_csv(
        manifest,
        OUTPUT_DIR / "split_manifest.csv",
    )

    print()
    print("Split generation complete.")

    print(
        f"Saved: {OUTPUT_DIR / 'train.csv'}"
    )

    print(
        f"Saved: {OUTPUT_DIR / 'val.csv'}"
    )

    print(
        f"Saved: {OUTPUT_DIR / 'test.csv'}"
    )

    print(
        f"Saved: {OUTPUT_DIR / 'split_manifest.csv'}"
    )


if __name__ == "__main__":
    main()