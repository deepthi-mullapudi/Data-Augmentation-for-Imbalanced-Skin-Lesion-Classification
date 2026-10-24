
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS_PATH = ROOT / "cache" / "splits.csv"
OUT_DIR = ROOT / "results" / "memorization_check_active"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

CANDIDATES = {
    "DCGAN-Aug-E150": ROOT / "results/fid_evaluation_dcgan_aug/dcgan_aug_epoch_150",
    "DCGAN-Aug-E200": ROOT / "results/fid_evaluation_dcgan_aug/dcgan_aug_epoch_200",
    "WGAN-GP-E150": ROOT / "results/fid_evaluation/wgangp_epoch_150",
    "WGAN-GP-E200": ROOT / "results/fid_evaluation/wgangp_epoch_200",
}

SEED = 42
SYNTHETIC_PER_CLASS = 200
BATCH_SIZE = 64
NUM_WORKERS = 0  # Safe default for Windows


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImagePathDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), str(path)


def index_real_images():
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    index = {}
    for path in DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_exts:
            index.setdefault(path.stem, path)
    return index


def extract_features(paths, model, device):
    """Return L2-normalized penultimate-layer ResNet-18 features."""
    loader = DataLoader(
        ImagePathDataset(paths),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    feature_batches = []
    ordered_paths = []

    with torch.inference_mode():
        for images, batch_paths in tqdm(loader, desc="Extracting features", leave=False):
            images = images.to(device, non_blocking=True)
            features = model(images)
            features = torch.flatten(features, start_dim=1)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            feature_batches.append(features.cpu().numpy())            
            ordered_paths.extend(batch_paths)

    if not feature_batches:
        return np.empty((0, 512), dtype=np.float32), []

    return np.concatenate(feature_batches).astype(np.float32), ordered_paths


def nearest_neighbors(query_features, reference_features):
    """Cosine distance to the nearest reference feature."""
    # Features are L2 normalized, so cosine similarity is the dot product.
    similarities = query_features @ reference_features.T
    nearest_indices = similarities.argmax(axis=1)
    nearest_similarities = similarities[
        np.arange(len(query_features)), nearest_indices
    ]
    distances = np.clip(1.0 - nearest_similarities, 0.0, 2.0)
    return distances, nearest_indices, nearest_similarities


def save_pair_grid(rows, output_path, max_pairs=24):
    """Save query/reference pairs, labelled with cosine distance."""
    if not rows:
        return

    chosen = rows[:max_pairs]
    thumb_size = 160
    label_h = 34
    gap = 8
    pair_w = thumb_size * 2 + gap
    canvas = Image.new(
        "RGB",
        (pair_w, len(chosen) * (thumb_size + label_h + gap)),
        "white",
    )

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for i, row in enumerate(chosen):
        y = i * (thumb_size + label_h + gap)
        for col, key in enumerate(("query_path", "nearest_path")):
            with Image.open(row[key]) as im:
                tile = im.convert("RGB").resize((thumb_size, thumb_size))
            canvas.paste(tile, (col * (thumb_size + gap), y))

        caption = (
            f"{row['source']} | {row['class']} | "
            f"cosine distance={row['distance']:.4f}"
        )
        draw.text((0, y + thumb_size + 2), caption, fill="black", font=font)

    canvas.save(output_path)


def main():
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = pd.read_csv(SPLITS_PATH)
    required = {"image_id", "dx", "split"}
    if not required.issubset(splits.columns):
        raise ValueError(f"splits.csv must contain {sorted(required)}")

    real_index = index_real_images()

    missing_ids = [
        image_id for image_id in splits["image_id"].astype(str)
        if image_id not in real_index
    ]
    if missing_ids:
        raise FileNotFoundError(
            f"{len(missing_ids)} images from splits.csv were not found. "
            f"Examples: {missing_ids[:5]}"
        )

    train = splits[splits["split"] == "train"].copy()
    val = splits[splits["split"] == "val"].copy()

    # No test images are used anywhere in this analysis.
    train_paths_by_class = {
        c: [real_index[x] for x in train.loc[train["dx"] == c, "image_id"].astype(str)]
        for c in CLASS_NAMES
    }
    val_paths_by_class = {
        c: [real_index[x] for x in val.loc[val["dx"] == c, "image_id"].astype(str)]
        for c in CLASS_NAMES
    }

    for candidate, folder in CANDIDATES.items():
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing candidate folder: {folder}")
        for class_name in CLASS_NAMES:
            class_dir = folder / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(class_dir)
            pngs = sorted(class_dir.glob("*.png"))
            if len(pngs) < SYNTHETIC_PER_CLASS:
                raise ValueError(
                    f"{candidate}/{class_name} has only {len(pngs)} PNG files"
                )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training reference images: {len(train)}")
    print(f"Validation baseline images: {len(val)}")
    print(f"Synthetic sample: {SYNTHETIC_PER_CLASS} per class per candidate")

    # ImageNet-pretrained ResNet-18; remove its classification head.
    weights = models.ResNet18_Weights.DEFAULT
    backbone = models.resnet18(weights=weights)
    model = torch.nn.Sequential(*list(backbone.children())[:-1])
    model = model.to(device).eval()

    # Extract real training features once per class.
    train_features = {}
    train_feature_paths = {}

    for class_name in CLASS_NAMES:
        feats, paths = extract_features(
            train_paths_by_class[class_name], model, device
        )
        train_features[class_name] = feats
        train_feature_paths[class_name] = paths
        print(f"Training features {class_name}: {len(paths)}")

    # Real-to-real baseline: each validation image vs same-class train images.
    records = []
    baseline_pair_rows = []

    for class_name in CLASS_NAMES:
        query_feats, query_paths = extract_features(
            val_paths_by_class[class_name], model, device
        )
        distances, indices, similarities = nearest_neighbors(
            query_feats, train_features[class_name]
        )

        for i, distance in enumerate(distances):
            nearest_path = train_feature_paths[class_name][indices[i]]
            records.append({
                "source": "real_validation",
                "candidate": "baseline",
                "class": class_name,
                "query_path": query_paths[i],
                "nearest_path": nearest_path,
                "distance": float(distance),
                "similarity": float(similarities[i]),
            })
            baseline_pair_rows.append({
                "source": "real_validation",
                "class": class_name,
                "query_path": query_paths[i],
                "nearest_path": nearest_path,
                "distance": float(distance),
            })

    # Synthetic-to-real training nearest neighbors, separately per class.
    for candidate, folder in CANDIDATES.items():
        print(f"\nAnalyzing {candidate}")

        for class_id, class_name in enumerate(CLASS_NAMES):
            all_pngs = sorted((folder / class_name).glob("*.png"))
            rng = random.Random(SEED + 1000 * class_id + len(records))
            selected_paths = sorted(
                rng.sample(all_pngs, SYNTHETIC_PER_CLASS)
            )

            query_feats, query_paths = extract_features(
                selected_paths, model, device
            )
            distances, indices, similarities = nearest_neighbors(
                query_feats, train_features[class_name]
            )

            for i, distance in enumerate(distances):
                nearest_path = train_feature_paths[class_name][indices[i]]
                records.append({
                    "source": "synthetic",
                    "candidate": candidate,
                    "class": class_name,
                    "query_path": query_paths[i],
                    "nearest_path": nearest_path,
                    "distance": float(distance),
                    "similarity": float(similarities[i]),
                })

            print(
                f"  {class_name}: "
                f"median distance={np.median(distances):.4f}, "
                f"mean distance={np.mean(distances):.4f}"
            )

    detailed = pd.DataFrame(records)
    detailed_path = OUT_DIR / "nearest_neighbor_distances.csv"
    detailed.to_csv(detailed_path, index=False)

    # Summary statistics by source, candidate and class.
    summary = (
        detailed.groupby(["source", "candidate", "class"])["distance"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    summary_path = OUT_DIR / "nearest_neighbor_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Plot per-class distributions, including validation baseline.
    for class_name in CLASS_NAMES:
        plt.figure(figsize=(11, 5))
        groups = []
        labels = []

        base = detailed[
            (detailed["source"] == "real_validation")
            & (detailed["class"] == class_name)
        ]["distance"].to_numpy()
        groups.append(base)
        labels.append("Real validation")

        for candidate in CANDIDATES:
            values = detailed[
                (detailed["source"] == "synthetic")
                & (detailed["candidate"] == candidate)
                & (detailed["class"] == class_name)
            ]["distance"].to_numpy()
            groups.append(values)
            labels.append(candidate)

        plt.boxplot(groups, tick_labels=labels, showfliers=False)
        plt.ylabel("Cosine distance to nearest same-class training image")
        plt.title(f"Nearest-neighbor distances: {class_name}")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"distance_distribution_{class_name}.png", dpi=160)
        plt.close()

    # Save examples with the smallest synthetic-to-training distances.
    synthetic = detailed[detailed["source"] == "synthetic"]
    for candidate in CANDIDATES:
        subset = synthetic[synthetic["candidate"] == candidate].sort_values(
            "distance"
        )
        save_pair_grid(
            subset.to_dict("records"),
            OUT_DIR / f"closest_pairs_{candidate}.png",
            max_pairs=24,
        )

    # Save closest real validation-to-training pairs as a comparison reference.
    baseline = detailed[detailed["source"] == "real_validation"].sort_values(
        "distance"
    )
    save_pair_grid(
        baseline.to_dict("records"),
        OUT_DIR / "closest_pairs_real_validation_baseline.png",
        max_pairs=24,
    )

    print("\nMemorization screening complete.")
    print(f"Detailed distances: {detailed_path}")
    print(f"Summary: {summary_path}")
    print(f"Plots and pair grids: {OUT_DIR}")


if __name__ == "__main__":
    main()
