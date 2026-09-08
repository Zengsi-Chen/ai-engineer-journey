from pathlib import Path
import random
import hashlib


def calculate_file_hash(path: Path) -> str:
    """
    Calculate SHA-256 hash of a file.
    """
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def find_duplicate_images(
    pairs: list[dict],
) -> dict[str, list[str]]:
    """
    Find duplicate images based on file content hash.
    """

    hash_map: dict[str, list[str]] = {}

    for pair in pairs:
        image_path = Path(pair["image_path"])
        file_hash = calculate_file_hash(image_path)

        hash_map.setdefault(file_hash, []).append(
            pair["sample_id"]
        )

    duplicates = {
        file_hash: sample_ids
        for file_hash, sample_ids in hash_map.items()
        if len(sample_ids) > 1
    }

    return duplicates


def build_image_mask_pairs(
    image_dir: Path,
    mask_dir: Path,
) -> list[dict]:
    """
    Build deterministic image-mask pairs based on matching file stems.
    """

    image_files = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    mask_files = sorted(
        path
        for path in mask_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    mask_map = {
        path.stem: path
        for path in mask_files
    }

    pairs = []

    for image_path in image_files:
        mask_path = mask_map.get(image_path.stem)

        if mask_path is None:
            continue

        pairs.append(
            {
                "sample_id": image_path.stem,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )

    return pairs


def split_pairs(
    pairs: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Create a deterministic train/validation/test split.
    """

    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 1.0"
        )

    if not pairs:
        raise ValueError("pairs cannot be empty")

    shuffled = pairs.copy()

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }