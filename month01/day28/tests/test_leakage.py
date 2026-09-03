from src.leakage_checks import (
    check_split_overlap,
    has_leakage,
)


def test_no_split_overlap():
    train_ids = {
        "image_001",
        "image_002",
        "image_003",
    }

    val_ids = {
        "image_004",
        "image_005",
    }

    test_ids = {
        "image_006",
        "image_007",
    }

    overlaps = check_split_overlap(
        train_ids,
        val_ids,
        test_ids,
    )

    assert overlaps["train_val"] == set()
    assert overlaps["train_test"] == set()
    assert overlaps["val_test"] == set()

    assert not has_leakage(overlaps)


def test_detect_train_validation_overlap():
    train_ids = {
        "image_001",
        "image_002",
        "image_003",
    }

    val_ids = {
        "image_003",
        "image_004",
    }

    test_ids = {
        "image_005",
    }

    overlaps = check_split_overlap(
        train_ids,
        val_ids,
        test_ids,
    )

    assert overlaps["train_val"] == {
        "image_003"
    }

    assert has_leakage(overlaps)


def test_detect_train_test_overlap():
    train_ids = {
        "image_001",
        "image_002",
    }

    val_ids = {
        "image_003",
    }

    test_ids = {
        "image_002",
        "image_004",
    }

    overlaps = check_split_overlap(
        train_ids,
        val_ids,
        test_ids,
    )

    assert overlaps["train_test"] == {
        "image_002"
    }

    assert has_leakage(overlaps)


def test_detect_validation_test_overlap():
    train_ids = {
        "image_001",
    }

    val_ids = {
        "image_002",
        "image_003",
    }

    test_ids = {
        "image_003",
        "image_004",
    }

    overlaps = check_split_overlap(
        train_ids,
        val_ids,
        test_ids,
    )

    assert overlaps["val_test"] == {
        "image_003"
    }

    assert has_leakage(overlaps)