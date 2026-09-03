from collections.abc import Iterable


LEAKAGE_TYPES = {
    "direct": "Validation/test samples are used during training.",
    "preprocessing": "Preprocessing is fit on non-training data.",
    "feature": "Features contain target or future information.",
    "augmentation": "Augmented versions of the same sample cross splits.",
    "duplicate": "Duplicate samples appear across splits.",
    "group": "The same group appears across splits.",
}


def check_split_overlap(
    train_ids: Iterable[str],
    val_ids: Iterable[str],
    test_ids: Iterable[str],
) -> dict[str, set[str]]:
    """
    Check whether sample IDs overlap across data splits.

    Returns:
        Dictionary containing overlapping sample IDs.
    """
    train_ids = set(train_ids)
    val_ids = set(val_ids)
    test_ids = set(test_ids)

    return {
        "train_val": train_ids & val_ids,
        "train_test": train_ids & test_ids,
        "val_test": val_ids & test_ids,
    }


def has_leakage(overlaps: dict[str, set[str]]) -> bool:
    """
    Return True if any split overlap exists.
    """
    return any(len(overlap) > 0 for overlap in overlaps.values())