from pathlib import Path

from medseg.data.validation import validate_dataset_structure


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "kvasir-seg" / "images"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "kvasir-seg" / "masks"


def test_kvasir_seg_integrity():
    result = validate_dataset_structure(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
    )

    assert result["image_count"] == 1000
    assert result["mask_count"] == 1000

    assert result["broken_images"] == []
    assert result["broken_masks"] == []

    assert result["missing_masks"] == []
    assert result["orphan_masks"] == []

    assert len(result["matched_pairs"]) == 1000