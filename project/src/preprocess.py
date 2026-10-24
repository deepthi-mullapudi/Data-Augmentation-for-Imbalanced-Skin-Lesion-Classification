
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"

SPLITS_PATH = CACHE_DIR / "splits.csv"
MAPPING_PATH = CACHE_DIR / "label_mapping.json"
OUTPUT_PATH = CACHE_DIR / "ham64.npz"

COUNTS_PATH = CACHE_DIR / "train_class_counts.csv"
GRID_PATH = CACHE_DIR / "train_real_grid.png"

IMAGE_SIZE = 64
SEED = 42

# Keep this fixed across every classifier experiment.
CLASSIFIER_INPUT_SIZE = 224

CLASS_ORDER = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
]


# ---------------------------------------------------------
# Find and index images
# ---------------------------------------------------------

def build_image_index():
    """Map each image ID to its source image path."""

    image_index = {}

    for path in DATA_DIR.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue

        image_id = path.stem

        if image_id in image_index:
            raise ValueError(
                f"Duplicate image ID found: {image_id}\n"
                f"{image_index[image_id]}\n{path}"
            )

        image_index[image_id] = path

    if not image_index:
        raise FileNotFoundError(
            f"No JPEG images found under {DATA_DIR}"
        )

    return image_index


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

def preprocess_image(image_path):
    """
    Read an image, convert BGR to RGB, center-crop to a
    square, and resize to 64x64 using INTER_AREA.

    Returns:
        uint8 NumPy array with shape (64, 64, 3).
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # OpenCV reads BGR; store RGB for correct display and use.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]
    crop_size = min(height, width)

    # Center crop. For a 600x450 image, this removes
    # 75 pixels from each side of the width.
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2

    image = image[
        top:top + crop_size,
        left:left + crop_size
    ]

    # INTER_AREA is suitable for downsampling.
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise RuntimeError(
            f"Unexpected image shape: {image.shape}"
        )

    return image.astype(np.uint8, copy=False)


# ---------------------------------------------------------
# Training counts
# ---------------------------------------------------------

def save_training_counts(df, label_map):
    train_df = df[df["split"] == "train"]

    counts = (
        train_df.groupby("dx")["image_id"]
        .nunique()
        .reindex(CLASS_ORDER, fill_value=0)
        .rename("train_images")
        .reset_index()
        .rename(columns={"index": "dx"})
    )

    counts["label_id"] = counts["dx"].map(label_map)
    counts = counts[["dx", "label_id", "train_images"]]

    counts.to_csv(COUNTS_PATH, index=False)

    print("\nTraining image counts:")
    print(counts.to_string(index=False))
    print(f"\nCounts saved to: {COUNTS_PATH}")

    return counts


# ---------------------------------------------------------
# Real training image grid
# ---------------------------------------------------------

def save_training_grid(images, df, label_map):
    """
    Save a grid with 7 class columns and 8 training images
    per class. This is for visual inspection only.
    """

    rng = np.random.default_rng(SEED)

    # Upscale each 64x64 image for convenient inspection.
    display_size = 192
    title_height = 42
    padding = 12

    canvas_width = (
        len(CLASS_ORDER) * (display_size + padding) + padding
    )
    canvas_height = (
        title_height
        + 8 * (display_size + padding)
        + padding
    )

    canvas = Image.new(
        "RGB", (canvas_width, canvas_height), "white"
    )
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for col, dx in enumerate(CLASS_ORDER):
        class_rows = df[
            (df["split"] == "train") & (df["dx"] == dx)
        ]

        if len(class_rows) < 8:
            raise ValueError(
                f"Class {dx} has fewer than 8 training images."
            )

        selected = rng.choice(
            class_rows.index.to_numpy(),
            size=8,
            replace=False
        )

        x = padding + col * (display_size + padding)

        draw.text(
            (x, 10),
            f"{dx} (ID {label_map[dx]})",
            fill="black",
            font=font
        )

        for row_num, row_idx in enumerate(selected):
            # Image array and dataframe use the same row order.
            image = images[row_idx]

            pil_image = Image.fromarray(image).resize(
                (display_size, display_size),
                Image.Resampling.LANCZOS
            )

            y = (
                title_height
                + padding
                + row_num * (display_size + padding)
            )

            canvas.paste(pil_image, (x, y))

    canvas.save(GRID_PATH)
    print(f"Training image grid saved to: {GRID_PATH}")


# ---------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SPLITS_PATH}. Create splits first."
        )

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MAPPING_PATH}. Create label mapping first."
        )

    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    if label_map != {
        "akiec": 0,
        "bcc": 1,
        "bkl": 2,
        "df": 3,
        "mel": 4,
        "nv": 5,
        "vasc": 6,
    }:
        raise ValueError(
            "Saved label mapping differs from the required mapping."
        )

    df = pd.read_csv(SPLITS_PATH)

    required_columns = {
        "image_id", "lesion_id", "dx", "split"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Split file must contain: {required_columns}"
        )

    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in splits.csv.")

    if df[["image_id", "lesion_id", "dx", "split"]].isna().any().any():
        raise ValueError("Missing values found in splits.csv.")

    if not set(df["dx"]).issubset(label_map):
        raise ValueError("Unknown class in splits.csv.")

    if not set(df["split"]).issubset({"train", "val", "test"}):
        raise ValueError("Unexpected split name found.")

    # Check lesion-level separation again before processing.
    lesion_sets = {
        split: set(df.loc[df["split"] == split, "lesion_id"])
        for split in ["train", "val", "test"]
    }

    assert lesion_sets["train"].isdisjoint(lesion_sets["val"])
    assert lesion_sets["train"].isdisjoint(lesion_sets["test"])
    assert lesion_sets["val"].isdisjoint(lesion_sets["test"])

    image_index = build_image_index()

    metadata_ids = set(df["image_id"])
    disk_ids = set(image_index)

    missing_images = metadata_ids - disk_ids
    extra_images = disk_ids - metadata_ids

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} images are missing. "
            f"Examples: {list(missing_images)[:10]}"
        )

    if extra_images:
        raise ValueError(
            f"{len(extra_images)} JPEGs do not appear in splits.csv. "
            f"Examples: {list(extra_images)[:10]}"
        )

    # The dataframe row order defines the order of all arrays.
    image_ids = df["image_id"].astype(str).to_numpy()
    labels = df["dx"].map(label_map).to_numpy(dtype=np.int64)
    splits = df["split"].astype(str).to_numpy()

    images = np.empty(
        (len(df), IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8
    )

    print(f"Preprocessing {len(df)} images...")

    for i, image_id in enumerate(image_ids):
        images[i] = preprocess_image(image_index[image_id])

        if (i + 1) % 1000 == 0 or i + 1 == len(df):
            print(f"Processed {i + 1}/{len(df)}")

    # Validate the completed cache before saving.
    assert images.shape == (len(df), 64, 64, 3)
    assert images.dtype == np.uint8
    assert labels.shape == (len(df),)
    assert splits.shape == (len(df),)

    if images.min() < 0 or images.max() > 255:
        raise ValueError("Unexpected pixel range.")

    # Keep image IDs so array positions remain traceable.
    np.savez_compressed(
    OUTPUT_PATH,
    images=images.astype(np.uint8, copy=False),
    labels=labels.astype(np.int64, copy=False),
    split=np.asarray(splits, dtype="U5"),
    image_ids=np.asarray(image_ids, dtype="U16"),
)

    print(f"\nCache saved to: {OUTPUT_PATH}")
    print(f"Image shape: {images.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Pixel range: {images.min()} to {images.max()}")
    print(f"Label shape: {labels.shape}")
    print(f"Split shape: {splits.shape}")
    print(f"Fixed classifier input size: {CLASSIFIER_INPUT_SIZE}")

    # Save counts and the real-image sanity grid.
    save_training_counts(df, label_map)
    save_training_grid(images, df, label_map)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
