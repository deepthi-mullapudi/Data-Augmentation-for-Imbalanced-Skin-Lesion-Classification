
from pathlib import Path
import random

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS_PATH = ROOT / "cache" / "splits.csv"
OUTPUT_DIR = ROOT / "results" / "visual_comparison_active"

CLASS_NAMES = ["akiec", "bcc", "df", "vasc"]

CANDIDATES = [
    ("DCGAN-Aug E150", ROOT / "results/fid_evaluation_dcgan_aug/dcgan_aug_epoch_150"),
    ("DCGAN-Aug E200", ROOT / "results/fid_evaluation_dcgan_aug/dcgan_aug_epoch_200"),
    ("WGAN-GP E150", ROOT / "results/fid_evaluation/wgangp_epoch_150"),
    ("WGAN-GP E200", ROOT / "results/fid_evaluation/wgangp_epoch_200"),
]

SEED = 42
IMAGES_PER_ROW = 8
TILE_SIZE = 128
LABEL_HEIGHT = 30
ROW_LABEL_WIDTH = 145
PADDING = 8


def index_real_images():
    """Map each real image ID to its file path."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    index = {}

    for path in DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            index.setdefault(path.stem, path)

    return index


def load_rgb(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return ImageOps.fit(
            image,
            (TILE_SIZE, TILE_SIZE),
            method=Image.Resampling.LANCZOS,
        )


def make_grid(class_name, real_paths, candidate_paths):
    """Create a five-row grid: real plus four GAN checkpoints."""
    rng = random.Random(SEED + CLASS_NAMES.index(class_name))

    rows = [("Real training images", real_paths)]

    for label, folder in CANDIDATES:
        image_folder = folder / class_name
        if not image_folder.is_dir():
            raise FileNotFoundError(image_folder)

        available = sorted(image_folder.glob("*.png"))
        if len(available) < IMAGES_PER_ROW:
            raise ValueError(
                f"Expected at least {IMAGES_PER_ROW} PNGs in {image_folder}; "
                f"found {len(available)}"
            )

        rows.append((label, available))

    # Select eight distinct samples from each source reproducibly.
    selected_rows = []
    for label, paths in rows:
        selected = rng.sample(paths, IMAGES_PER_ROW)
        selected_rows.append((label, selected))

    grid_width = (
        ROW_LABEL_WIDTH
        + IMAGES_PER_ROW * TILE_SIZE
        + (IMAGES_PER_ROW - 1) * PADDING
        + 2 * PADDING
    )
    grid_height = (
        len(selected_rows) * (TILE_SIZE + LABEL_HEIGHT + PADDING)
        + PADDING
    )

    canvas = Image.new("RGB", (grid_width, grid_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for row_index, (label, paths) in enumerate(selected_rows):
        y = PADDING + row_index * (TILE_SIZE + LABEL_HEIGHT + PADDING)

        draw.text(
            (PADDING, y + 5),
            label,
            fill="black",
            font=font,
        )

        for col_index, path in enumerate(paths):
            x = (
                ROW_LABEL_WIDTH
                + PADDING
                + col_index * (TILE_SIZE + PADDING)
            )

            tile = load_rgb(path)
            canvas.paste(tile, (x, y))

            # Thin border helps separate adjacent samples.
            draw.rectangle(
                (x, y, x + TILE_SIZE - 1, y + TILE_SIZE - 1),
                outline=(180, 180, 180),
                width=1,
            )

    output_path = OUTPUT_DIR / f"{class_name}_comparison.png"
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SPLITS_PATH.exists():
        raise FileNotFoundError(SPLITS_PATH)

    splits = pd.read_csv(SPLITS_PATH)

    required_columns = {"image_id", "dx", "split"}
    if not required_columns.issubset(splits.columns):
        raise ValueError(
            f"splits.csv must contain {sorted(required_columns)}"
        )

    # Crucially, real images come only from the fixed training split.
    train = splits.loc[
        splits["split"] == "train",
        ["image_id", "dx"],
    ].drop_duplicates(subset="image_id")

    image_index = index_real_images()
    print(f"Indexed {len(image_index)} real image files.")
    print(f"Training-split records: {len(train)}")

    for class_name in CLASS_NAMES:
        class_rows = train.loc[train["dx"] == class_name]
        if len(class_rows) < IMAGES_PER_ROW:
            raise ValueError(
                f"Not enough training images for class {class_name}"
            )

        missing = [
            image_id for image_id in class_rows["image_id"]
            if image_id not in image_index
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} real images missing for {class_name}. "
                f"Examples: {missing[:5]}"
            )

        real_paths = [
            image_index[image_id]
            for image_id in class_rows["image_id"]
        ]

        make_grid(class_name, real_paths, CANDIDATES)

    print("\nAll comparison grids created.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
