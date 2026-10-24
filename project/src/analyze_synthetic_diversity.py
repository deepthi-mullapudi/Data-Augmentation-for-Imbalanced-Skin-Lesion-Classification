from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

# Active candidates only:
# - Revised DCGAN (DCGAN-Aug), replacing the archived original DCGAN
# - WGAN-GP
CANDIDATES = {
    "dcgan_aug_epoch_150": (
        ROOT / "results" / "fid_evaluation_dcgan_aug" / "dcgan_aug_epoch_150"
    ),
    "dcgan_aug_epoch_200": (
        ROOT / "results" / "fid_evaluation_dcgan_aug" / "dcgan_aug_epoch_200"
    ),
    "wgangp_epoch_150": (
        ROOT / "results" / "fid_evaluation" / "wgangp_epoch_150"
    ),
    "wgangp_epoch_200": (
        ROOT / "results" / "fid_evaluation" / "wgangp_epoch_200"
    ),
}

# Use the same fixed number of images per class for every candidate.
# 100 images/class matches the scale of the earlier preliminary analysis
# and avoids the large computational cost of all pairs among 1,000 images.
IMAGES_PER_CLASS = 100
SAMPLING_SEED = 2026

OUTPUT_DIR = ROOT / "results" / "synthetic_diversity_active"
OUTPUT_CSV = OUTPUT_DIR / "diversity_metrics.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "diversity_summary.csv"


def normalized_pixel_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute RGB pixel difference, normalized to [0, 1]."""
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    return float(np.mean(np.abs(a - b)))


def load_images(class_dir: Path) -> list[np.ndarray]:
    images = []
    for path in sorted(class_dir.glob("*.png")):
        with Image.open(path) as im:
            images.append(np.asarray(im.convert("RGB")))
    return images


def choose_images(images: list[np.ndarray], count: int, seed: int) -> list[np.ndarray]:
    """Choose a reproducible subset without replacement."""
    if len(images) <= count:
        return images
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(images), size=count, replace=False))
    return [images[i] for i in indices]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [
        f"{name}: {path}"
        for name, path in CANDIDATES.items()
        if not path.is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "The following active candidate directories were not found:\n"
            + "\n".join(missing)
            + "\nCheck the paths before running the analysis."
        )

    rows = []

    for candidate_name, candidate_dir in CANDIDATES.items():
        print(f"\n=== {candidate_name} ===")
        candidate_class_means = []

        class_dirs = sorted(
            p for p in candidate_dir.iterdir()
            if p.is_dir() and any(p.glob("*.png"))
        )
        if not class_dirs:
            raise RuntimeError(f"No class directories with PNGs found: {candidate_dir}")

        for class_dir in class_dirs:
            images_all = load_images(class_dir)
            if len(images_all) < 2:
                print(f"Skipping {class_dir.name}: fewer than 2 images.")
                continue

            class_seed = SAMPLING_SEED + sum(ord(ch) for ch in class_dir.name)
            images = choose_images(images_all, IMAGES_PER_CLASS, class_seed)

            distances = [
                normalized_pixel_distance(images[i], images[j])
                for i, j in combinations(range(len(images)), 2)
            ]

            mean_distance = float(np.mean(distances))
            median_distance = float(np.median(distances))
            min_distance = float(np.min(distances))
            max_distance = float(np.max(distances))

            rows.append({
                "checkpoint": candidate_name,
                "gan": "DCGAN-Aug" if candidate_name.startswith("dcgan_aug_") else "WGAN-GP",
                "epoch": int(candidate_name.rsplit("_", 1)[-1]),
                "class_name": class_dir.name,
                "num_images_available": len(images_all),
                "num_images_used": len(images),
                "num_pairs": len(distances),
                "mean_pairwise_distance": mean_distance,
                "median_pairwise_distance": median_distance,
                "min_pairwise_distance": min_distance,
                "max_pairwise_distance": max_distance,
                "metric": "normalized raw-pixel mean absolute pairwise distance",
                "sampling_seed": class_seed,
            })

            candidate_class_means.append(mean_distance)
            print(
                f"{class_dir.name}: available={len(images_all)}, "
                f"used={len(images)}, pairs={len(distances)}, "
                f"mean={mean_distance:.4f}, median={median_distance:.4f}, "
                f"min={min_distance:.4f}"
            )

        if candidate_class_means:
            print(
                "Unweighted mean of class means:",
                f"{np.mean(candidate_class_means):.4f}"
            )

    if not rows:
        raise RuntimeError("No diversity measurements were produced.")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    summary = (
        df.groupby(["checkpoint", "gan", "epoch"], as_index=False)
        .agg(
            classes_evaluated=("class_name", "nunique"),
            images_per_class_used=("num_images_used", "min"),
            mean_of_class_means=("mean_pairwise_distance", "mean"),
            std_of_class_means=("mean_pairwise_distance", "std"),
        )
        .sort_values(["gan", "epoch"])
    )
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(f"\nSaved per-class diversity metrics to:\n{OUTPUT_CSV}")
    print(f"Saved checkpoint summary to:\n{OUTPUT_SUMMARY_CSV}")
    print("\nSummary by checkpoint:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
