
'''
The script is designed to create:
- results/fid_evaluation_active/fid_results_table.csv — comparison table with seven classes plus overall, across four active checkpoints.
- results/fid_evaluation_active/fid_results_long.csv — detailed results, including real and synthetic sample counts.
'''

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch_fidelity import calculate_metrics


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS_PATH = ROOT / "cache" / "splits.csv"
METADATA_PATH = DATA_DIR / "HAM10000_metadata"
OUTPUT_DIR = ROOT / "results" / "fid_evaluation_active"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# Only active experiments. Do not add the archived original DCGAN here.
CANDIDATES = {
    "DCGAN-Aug-E150": (
        ROOT / "results/fid_evaluation_dcgan_aug/dcgan_aug_epoch_150"
    ),
    "DCGAN-Aug-E200": (
        ROOT / "results/fid_evaluation_dcgan_aug/dcgan_aug_epoch_200"
    ),
    "WGAN-GP-E150": (
        ROOT / "results/fid_evaluation/wgangp_epoch_150"
    ),
    "WGAN-GP-E200": (
        ROOT / "results/fid_evaluation/wgangp_epoch_200"
    ),
}


def find_image_files():
    """Index image files by filename stem."""
    index = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for path in DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_extensions:
            index.setdefault(path.stem, path)

    return index


def load_npz(npz_path):
    """Find the image and label arrays without assuming their key names."""
    with np.load(npz_path, allow_pickle=False) as data:
        keys = list(data.files)

        image_keys = [
            k for k in keys
            if data[k].ndim == 4 and data[k].shape[-1] in (1, 3, 4)
        ]
        if len(image_keys) != 1:
            raise ValueError(
                f"Could not uniquely identify image array in {npz_path}. "
                f"Keys and shapes: {[(k, data[k].shape) for k in keys]}"
            )

        image_key = image_keys[0]
        images = data[image_key]
        n = len(images)

        label_keys = [
            k for k in keys
            if data[k].ndim == 1 and len(data[k]) == n
            and k != image_key
        ]
        if len(label_keys) != 1:
            raise ValueError(
                f"Could not uniquely identify label array in {npz_path}. "
                f"Keys and shapes: {[(k, data[k].shape) for k in keys]}"
            )

        labels = data[label_keys[0]]

    # The project's generated NPZ files should contain uint8 images.
    if images.dtype != np.uint8:
        if np.issubdtype(images.dtype, np.floating):
            lo, hi = float(images.min()), float(images.max())
            if lo >= 0 and hi <= 1:
                images = images * 255.0
            elif lo < 0 and hi <= 1:
                images = (images + 1.0) * 127.5
            else:
                raise ValueError(
                    f"Unexpected floating-point image range: {lo} to {hi}"
                )
            images = np.clip(images, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected image dtype: {images.dtype}")

    labels = np.asarray(labels)

    # Accept either integer labels 0–6 or class-name labels.
    if labels.dtype.kind in {"U", "S", "O"}:
        labels = np.array([
            LABEL_MAP[str(x)] for x in labels
        ], dtype=np.int64)
    else:
        labels = labels.astype(np.int64)

    if set(np.unique(labels)) - set(range(len(CLASS_NAMES))):
        raise ValueError(f"Unexpected class labels in {npz_path}")

    if images.shape[1:3] != (64, 64):
        raise ValueError(
            f"Expected 64x64 images in {npz_path}; got {images.shape}"
        )

    return images, labels


def write_images(images, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        if image.shape[-1] == 1:
            image = image[:, :, 0]
            pil_image = Image.fromarray(image, mode="L").convert("RGB")
        else:
            pil_image = Image.fromarray(image[:, :, :3], mode="RGB")
        pil_image.save(destination / f"{i:06d}.png")


def run_fid(real_dir, synthetic_dir):
    metrics = calculate_metrics(
        input1=str(real_dir),
        input2=str(synthetic_dir),
        cuda=torch.cuda.is_available(),
        fid=True,
        isc=False,
        kid=False,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SPLITS_PATH.exists():
        raise FileNotFoundError(SPLITS_PATH)
    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    splits = pd.read_csv(SPLITS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    # Use the fixed split file as the source of truth.
    train = splits.loc[splits["split"] == "train", ["image_id", "dx"]].copy()
    train = train.drop_duplicates(subset="image_id")

    # Validate labels against the metadata.
    merged = train.merge(
        metadata[["image_id", "dx"]],
        on="image_id",
        how="left",
        suffixes=("_split", "_metadata"),
        validate="one_to_one",
    )
    if merged["dx_metadata"].isna().any():
        raise ValueError("Some training image IDs are missing from metadata.")
    if (merged["dx_split"] != merged["dx_metadata"]).any():
        raise ValueError("Labels disagree between splits.csv and metadata.")

    train["dx"] = train["dx"].astype(str)
    if not set(train["dx"]).issubset(set(CLASS_NAMES)):
        raise ValueError("Unexpected class name in training split.")

    image_index = find_image_files()
    missing = [image_id for image_id in train["image_id"]
               if image_id not in image_index]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} training images were not found. "
            f"Examples: {missing[:5]}"
        )

    # Validate every candidate before starting the expensive FID calculations.
    candidate_data = {}
    for candidate, folder in CANDIDATES.items():
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing candidate directory: {folder}")

        npz_files = list(folder.glob("*.npz"))
        if len(npz_files) != 1:
            raise ValueError(
                f"Expected exactly one NPZ in {folder}, found: {npz_files}"
            )

        images, labels = load_npz(npz_files[0])
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        if not np.all(counts == 1000):
            raise ValueError(
                f"{candidate} should contain 1000 images per class; "
                f"found {dict(zip(CLASS_NAMES, counts.tolist()))}"
            )

        candidate_data[candidate] = (images, labels)
        print(
            f"Validated {candidate}: {len(images)} images; "
            f"NPZ={npz_files[0].name}"
        )

    print(f"Real training images: {len(train)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Preparing reference images and calculating FID...")

    records = []

    with tempfile.TemporaryDirectory(prefix="ham_fid_") as temp:
        temp_root = Path(temp)
        real_root = temp_root / "real"
        synthetic_root = temp_root / "synthetic"

        # Prepare the real training reference set once.
        real_all = real_root / "overall"
        write_images(
            np.stack([
                np.asarray(Image.open(image_index[row.image_id]).convert("RGB"))
                for row in train.itertuples()
            ]),
            real_all,
        )

        for class_name in CLASS_NAMES:
            class_rows = train.loc[train["dx"] == class_name]
            class_dir = real_root / class_name

            write_images(
                np.stack([
                    np.asarray(
                        Image.open(image_index[row.image_id]).convert("RGB")
                    )
                    for row in class_rows.itertuples()
                ]),
                class_dir,
            )
            print(f"Prepared real {class_name}: {len(class_rows)} images")

        for candidate, (images, labels) in candidate_data.items():
            print(f"\nEvaluating {candidate}...")

            candidate_root = synthetic_root / candidate
            write_images(images, candidate_root / "overall")

            for class_id, class_name in enumerate(CLASS_NAMES):
                class_images = images[labels == class_id]
                write_images(class_images, candidate_root / class_name)

            # Overall FID: all synthetic images vs all real training images.
            overall_fid = run_fid(
                real_root / "overall",
                candidate_root / "overall",
            )
            records.append({
                "candidate": candidate,
                "class": "overall",
                "real_count": len(train),
                "synthetic_count": len(images),
                "fid": overall_fid,
            })
            print(f"  Overall FID: {overall_fid:.4f}")

            # Class-conditional FID: each class vs its real training images.
            for class_name in CLASS_NAMES:
                real_count = int((train["dx"] == class_name).sum())
                synthetic_count = int((labels == LABEL_MAP[class_name]).sum())

                fid = run_fid(
                    real_root / class_name,
                    candidate_root / class_name,
                )
                records.append({
                    "candidate": candidate,
                    "class": class_name,
                    "real_count": real_count,
                    "synthetic_count": synthetic_count,
                    "fid": fid,
                })
                print(f"  {class_name}: FID={fid:.4f}")

    results = pd.DataFrame(records)
    long_path = OUTPUT_DIR / "fid_results_long.csv"
    wide_path = OUTPUT_DIR / "fid_results_table.csv"

    results.to_csv(long_path, index=False)

    # One row per class plus overall; one column per active checkpoint.
    wide = results.pivot(index="class", columns="candidate", values="fid")
    wide = wide.reindex(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "overall"])
    wide.to_csv(wide_path)

    print("\nFID evaluation complete.")
    print(f"Detailed results: {long_path}")
    print(f"Comparison table: {wide_path}")
    print("\nFID comparison table:")
    print(wide.round(4).to_string())


if __name__ == "__main__":
    main()
