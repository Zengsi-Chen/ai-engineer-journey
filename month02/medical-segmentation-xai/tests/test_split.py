import csv
import hashlib

from pathlib import Path

from medseg.data.split import (
    build_image_mask_pairs,
    split_pairs,
    find_duplicate_images,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

SPLIT_DIR = PROJECT_ROOT / "artifacts" / "splits"

def test_build_image_mask_pairs():
    pairs = build_image_mask_pairs(
        IMAGE_DIR,
        MASK_DIR,
    )

    assert len(pairs) == 1000

    sample_ids = [
        pair["sample_id"]
        for pair in pairs
    ]

    assert len(set(sample_ids)) == 1000


def test_split_counts():
    pairs = build_image_mask_pairs(
        IMAGE_DIR,
        MASK_DIR,
    )

    splits = split_pairs(
        pairs,
        seed=42,
    )

    assert len(splits["train"]) == 700
    assert len(splits["val"]) == 150
    assert len(splits["test"]) == 150


def test_split_has_no_overlap():
    pairs = build_image_mask_pairs(
        IMAGE_DIR,
        MASK_DIR,
    )

    splits = split_pairs(
        pairs,
        seed=42,
    )

    train_ids = {
        item["sample_id"]
        for item in splits["train"]
    }

    val_ids = {
        item["sample_id"]
        for item in splits["val"]
    }

    test_ids = {
        item["sample_id"]
        for item in splits["test"]
    }

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_full_coverage():
    pairs = build_image_mask_pairs(
        IMAGE_DIR,
        MASK_DIR,
    )

    splits = split_pairs(
        pairs,
        seed=42,
    )

    all_split_ids = (
        {
            item["sample_id"]
            for item in splits["train"]
        }
        | {
            item["sample_id"]
            for item in splits["val"]
        }
        | {
            item["sample_id"]
            for item in splits["test"]
        }
    )

    original_ids = {
        item["sample_id"]
        for item in pairs
    }

    assert all_split_ids == original_ids


def test_split_is_deterministic():
    pairs = build_image_mask_pairs(
        IMAGE_DIR,
        MASK_DIR,
    )

    split_a = split_pairs(
        pairs,
        seed=42,
    )

    split_b = split_pairs(
        pairs,
        seed=42,
    )

    for split_name in ["train", "val", "test"]:
        ids_a = [
            item["sample_id"]
            for item in split_a[split_name]
        ]

        ids_b = [
            item["sample_id"]
            for item in split_b[split_name]
        ]

        assert ids_a == ids_b


def test_no_duplicate_images():

    pairs = build_image_mask_pairs(
        IMAGE_DIR,
        MASK_DIR,
    )

    duplicates = find_duplicate_images(pairs)

    assert duplicates == {}


def calculate_test_hash(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_manifest():
    manifest_path = SPLIT_DIR / "split_manifest.csv"

    assert manifest_path.exists(), (
        f"Missing split manifest: {manifest_path}"
    )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def test_manifest_exists():
    manifest_path = SPLIT_DIR / "split_manifest.csv"

    assert manifest_path.exists()


def test_manifest_counts():
    rows = load_manifest()

    assert len(rows) == 1000

    counts = {}

    for row in rows:
        split = row["split"]
        counts[split] = counts.get(split, 0) + 1

    assert counts == {
        "train": 700,
        "val": 150,
        "test": 150,
    }


def test_manifest_sample_ids_are_unique():
    rows = load_manifest()

    sample_ids = [
        row["sample_id"]
        for row in rows
    ]

    assert len(sample_ids) == len(set(sample_ids))


def test_manifest_paths_exist():
    rows = load_manifest()

    for row in rows:
        image_path = PROJECT_ROOT / row["image_path"]
        mask_path = PROJECT_ROOT / row["mask_path"]

        assert image_path.exists(), (
            f"Missing image: {image_path}"
        )

        assert mask_path.exists(), (
            f"Missing mask: {mask_path}"
        )


def test_manifest_split_disjoint():
    rows = load_manifest()

    split_ids = {
        "train": set(),
        "val": set(),
        "test": set(),
    }

    for row in rows:
        split_ids[row["split"]].add(
            row["sample_id"]
        )

    assert split_ids["train"].isdisjoint(
        split_ids["val"]
    )

    assert split_ids["train"].isdisjoint(
        split_ids["test"]
    )

    assert split_ids["val"].isdisjoint(
        split_ids["test"]
    )


def test_manifest_hashes_do_not_cross_splits():
    rows = load_manifest()

    split_hashes = {
        "train": set(),
        "val": set(),
        "test": set(),
    }

    for row in rows:
        image_path = PROJECT_ROOT / row["image_path"]

        file_hash = calculate_test_hash(
            image_path
        )

        split_hashes[row["split"]].add(
            file_hash
        )

    assert split_hashes["train"].isdisjoint(
        split_hashes["val"]
    )

    assert split_hashes["train"].isdisjoint(
        split_hashes["test"]
    )

    assert split_hashes["val"].isdisjoint(
        split_hashes["test"]
    )
