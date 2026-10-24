# """HAM10000 loading, splitting, preprocessing, and dataset utilities."""

# from pathlib import Path
# import json

# import pandas as pd
# from sklearn.model_selection import StratifiedGroupKFold


# # Project paths
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# DATA_DIR = PROJECT_ROOT / "data"
# CACHE_DIR = PROJECT_ROOT / "cache"

# CACHE_DIR.mkdir(parents=True, exist_ok=True)

# # Fixed label IDs shared by the GAN and classifier
# LABEL_MAP = {
#     "akiec": 0,
#     "bcc": 1,
#     "bkl": 2,
#     "df": 3,
#     "mel": 4,
#     "nv": 5,
#     "vasc": 6,
# }


# def find_metadata():
#     candidates = [
#         p for p in DATA_DIR.rglob("HAM10000_metadata*")
#         if p.is_file()
#     ]

#     if len(candidates) != 1:
#         raise FileNotFoundError(
#             f"Expected one HAM10000 metadata file, "
#             f"but found {len(candidates)}: {candidates}"
#         )

#     return candidates[0]


# def create_splits(seed=42):
#     metadata_path = find_metadata()
#     df = pd.read_csv(metadata_path)

#     required = {"image_id", "lesion_id", "dx"}
#     missing_columns = required - set(df.columns)
#     if missing_columns:
#         raise ValueError(f"Missing columns: {missing_columns}")

#     if df[list(required)].isna().any().any():
#         raise ValueError("Missing values found in required columns.")

#     if df["image_id"].duplicated().any():
#         raise ValueError("Duplicate image_id values found.")

#     unknown_labels = set(df["dx"].unique()) - set(LABEL_MAP)
#     if unknown_labels:
#         raise ValueError(f"Unknown diagnostic labels: {unknown_labels}")

#     # A lesion must not have conflicting diagnostic labels.
#     labels_per_lesion = df.groupby("lesion_id")["dx"].nunique()
#     if (labels_per_lesion > 1).any():
#         raise ValueError(
#             "Some lesion_id values have conflicting dx labels."
#         )

#     # Persist the mapping used by every experiment.
#     mapping_path = CACHE_DIR / "label_mapping.json"
#     with mapping_path.open("w", encoding="utf-8") as f:
#         json.dump(LABEL_MAP, f, indent=2)

#     # Assign each image its fixed integer label.
#     y = df["dx"].map(LABEL_MAP).to_numpy()
#     groups = df["lesion_id"].to_numpy()

#     # Twenty grouped folds; combine folds into approximately
#     # 70% train, 15% validation and 15% test.
#     splitter = StratifiedGroupKFold(
#         n_splits=20,
#         shuffle=True,
#         random_state=seed,
#     )

#     fold_ids = [-1] * len(df)

#     for fold_id, (_, heldout_indices) in enumerate(
#         splitter.split(df, y, groups)
#     ):
#         for idx in heldout_indices:
#             fold_ids[idx] = fold_id

#     if any(fold_id == -1 for fold_id in fold_ids):
#         raise RuntimeError("Some images were not assigned a fold.")

#     df["fold"] = fold_ids

#     # Use fixed folds so the result is reproducible.
#     df["split"] = "train"
#     df.loc[df["fold"].isin([0, 1, 2]), "split"] = "test"
#     df.loc[df["fold"].isin([3, 4, 5]), "split"] = "val"

#     # Verify lesion-level isolation.
#     lesion_sets = {
#         name: set(df.loc[df["split"] == name, "lesion_id"])
#         for name in ["train", "val", "test"]
#     }

#     assert lesion_sets["train"].isdisjoint(lesion_sets["val"])
#     assert lesion_sets["train"].isdisjoint(lesion_sets["test"])
#     assert lesion_sets["val"].isdisjoint(lesion_sets["test"])

#     # Every class must occur in all three splits.
#     expected_classes = set(LABEL_MAP)
#     for split_name in ["train", "val", "test"]:
#         present = set(df.loc[df["split"] == split_name, "dx"])
#         missing = expected_classes - present
#         if missing:
#             raise ValueError(
#                 f"{split_name} is missing classes: {sorted(missing)}. "
#                 "Inspect lesion-level class counts before proceeding."
#             )

#     # Save the requested split manifest.
#     output_path = CACHE_DIR / "splits.csv"
#     df[["image_id", "lesion_id", "dx", "split"]].to_csv(
#         output_path, index=False
#     )

#     # Print split sizes and class distributions.
#     print(f"\nMetadata: {metadata_path}")
#     print(f"Saved mapping: {mapping_path}")
#     print(f"Saved splits: {output_path}")

#     print("\nImage counts per split:")
#     print(df["split"].value_counts())

#     print("\nClass counts per split:")
#     print(
#         pd.crosstab(df["split"], df["dx"])
#         [["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]]
#     )

#     print("\nClass proportions within each split:")
#     print(
#         pd.crosstab(df["split"], df["dx"], normalize="index")
#         [["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]]
#         .round(4)
#     )

#     print("\nUnique lesions per split:")
#     for name in ["train", "val", "test"]:
#         print(f"{name}: {len(lesion_sets[name])}")

#     print("\nNo lesion leakage detected.")
#     print("All seven classes appear in all three splits.")


# if __name__ == "__main__":
#     create_splits(seed=42)
    
    
"""HAM10000 loading, splitting, and classifier dataset utilities."""
from pathlib import Path
import json
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6,
}
CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda item: item[1])]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_metadata():
    candidates = [p for p in DATA_DIR.rglob("HAM10000_metadata*") if p.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one HAM10000 metadata file, but found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def create_splits(seed=42):
    """Create the fixed lesion-grouped split manifest. Do not rerun once finalized."""
    metadata_path = find_metadata()
    df = pd.read_csv(metadata_path)
    required = {"image_id", "lesion_id", "dx"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    if df[list(required)].isna().any().any():
        raise ValueError("Missing values found in required columns.")
    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values found.")
    unknown_labels = set(df["dx"].unique()) - set(LABEL_MAP)
    if unknown_labels:
        raise ValueError(f"Unknown diagnostic labels: {unknown_labels}")
    if (df.groupby("lesion_id")["dx"].nunique() > 1).any():
        raise ValueError("Some lesion_id values have conflicting dx labels.")

    mapping_path = CACHE_DIR / "label_mapping.json"
    mapping_path.write_text(json.dumps(LABEL_MAP, indent=2), encoding="utf-8")
    y = df["dx"].map(LABEL_MAP).to_numpy()
    groups = df["lesion_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold_ids = np.full(len(df), -1, dtype=np.int64)
    for fold_id, (_, heldout_indices) in enumerate(splitter.split(df, y, groups)):
        fold_ids[heldout_indices] = fold_id
    if np.any(fold_ids == -1):
        raise RuntimeError("Some images were not assigned a fold.")

    df["fold"] = fold_ids
    df["split"] = "train"
    df.loc[df["fold"].isin([0, 1, 2]), "split"] = "test"
    df.loc[df["fold"].isin([3, 4, 5]), "split"] = "val"

    lesion_sets = {
        name: set(df.loc[df["split"] == name, "lesion_id"])
        for name in ("train", "val", "test")
    }
    assert lesion_sets["train"].isdisjoint(lesion_sets["val"])
    assert lesion_sets["train"].isdisjoint(lesion_sets["test"])
    assert lesion_sets["val"].isdisjoint(lesion_sets["test"])

    for split_name in ("train", "val", "test"):
        present = set(df.loc[df["split"] == split_name, "dx"])
        missing = set(LABEL_MAP) - present
        if missing:
            raise ValueError(f"{split_name} is missing classes: {sorted(missing)}")

    output_path = CACHE_DIR / "splits.csv"
    df[["image_id", "lesion_id", "dx", "split"]].to_csv(output_path, index=False)
    print(f"Metadata: {metadata_path}")
    print(f"Saved mapping: {mapping_path}")
    print(f"Saved splits: {output_path}")
    print("\nImage counts per split:\n", df["split"].value_counts())
    print("\nClass counts per split:\n", pd.crosstab(df["split"], df["dx"])[CLASS_NAMES])
    print("\nUnique lesions per split:")
    for name in ("train", "val", "test"):
        print(f"{name}: {len(lesion_sets[name])}")
    print("No lesion leakage detected. All seven classes appear in all three splits.")


def _build_transform(split: str, image_size: int, augmentation: str):
    ops = [transforms.Resize((image_size, image_size))]
    if split == "train" and augmentation == "classical":
        # Applied only to the real/synthetic training dataset in E2/E3/E4
        # when the experiment config explicitly requests classical augmentation.
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(ops)


class HAMClassifierDataset(Dataset):
    """Dataset returning (normalized image tensor, integer label).

    Args:
        cache_path: path to cache/ham64.npz.
        split_csv: path to cache/splits.csv.
        split: one of 'train', 'val', 'test'.
        image_size: classifier input size (normally 224).
        augmentation: 'none' or 'classical'; classical transforms are train-only.
        extra_pairs: optional sequence of (image, label) pairs, appended to this
            dataset. Images may be HxWx3 uint8 arrays, PIL images, or paths.
            Extra pairs are intended for synthetic training images only.
        max_items: optional limit for smoke tests; applies to real examples before
            extra_pairs are appended.
    """
    def __init__(
        self,
        cache_path="cache/ham64.npz",
        split_csv="cache/splits.csv",
        split="train",
        image_size=224,
        augmentation="none",
        extra_pairs: Optional[Sequence[Tuple[object, int]]] = None,
        max_items: Optional[int] = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if augmentation not in {"none", "classical"}:
            raise ValueError("augmentation must be 'none' or 'classical'")
        if split != "train" and (augmentation != "none" or extra_pairs):
            raise ValueError("Augmentation and synthetic extra_pairs are training-only.")

        cache_path = Path(cache_path)
        split_csv = Path(split_csv)
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        if not split_csv.is_absolute():
            split_csv = PROJECT_ROOT / split_csv

        with np.load(cache_path, allow_pickle=False) as cache:
            all_images = cache["images"]
            all_labels = cache["labels"]
            all_splits = cache["split"]
            all_image_ids = cache["image_ids"]

        manifest = pd.read_csv(split_csv)
        if len(manifest) != len(all_images):
            raise ValueError(f"Cache/manifest length mismatch: {len(all_images)} vs {len(manifest)}")
        if not np.array_equal(all_image_ids.astype(str), manifest["image_id"].astype(str).to_numpy()):
            raise ValueError("Cache image_ids do not align with split manifest row order.")
        if not np.array_equal(all_splits.astype(str), manifest["split"].astype(str).to_numpy()):
            raise ValueError("Cache split array does not align with split manifest.")
        expected_labels = manifest["dx"].map(LABEL_MAP).to_numpy()
        if not np.array_equal(all_labels, expected_labels):
            raise ValueError("Cache labels do not align with manifest diagnostic labels.")

        mask = all_splits.astype(str) == split
        self.images = all_images[mask]
        self.labels = all_labels[mask].astype(np.int64)
        self.image_ids = all_image_ids[mask].astype(str).tolist()
        if max_items is not None:
            if max_items < 1:
                raise ValueError("max_items must be positive.")
            if len(self.labels) > max_items:
                # Deterministic class-balanced subset for smoke tests. This ensures
                # every class is represented in the small validation/test subset
                # so macro OvR AUC remains defined.
                selected = []
                per_class = {
                    class_id: np.flatnonzero(self.labels == class_id).tolist()
                    for class_id in range(len(LABEL_MAP))
                }
                offsets = {class_id: 0 for class_id in per_class}
                while len(selected) < max_items:
                    made_progress = False
                    for class_id in range(len(LABEL_MAP)):
                        candidates = per_class[class_id]
                        pos = offsets[class_id]
                        if pos < len(candidates) and len(selected) < max_items:
                            selected.append(candidates[pos])
                            offsets[class_id] += 1
                            made_progress = True
                    if not made_progress:
                        break
                selected = np.asarray(selected, dtype=np.int64)
                self.images = self.images[selected]
                self.labels = self.labels[selected]
                self.image_ids = [self.image_ids[i] for i in selected]

        self.transform = _build_transform(split, image_size, augmentation)
        self.extra_pairs = list(extra_pairs or [])
        if self.extra_pairs and split != "train":
            raise ValueError("Synthetic examples may only be appended to training data.")
        for _, label in self.extra_pairs:
            if int(label) not in range(len(LABEL_MAP)):
                raise ValueError(f"Invalid synthetic label: {label}")

    def __len__(self):
        return len(self.labels) + len(self.extra_pairs)

    def _to_pil(self, image):
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            # Float synthetic images are accepted if in [0,1] or [-1,1].
            arr = arr.astype(np.float32)
            if arr.min() < 0:
                arr = (arr + 1.0) / 2.0
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def __getitem__(self, index):
        if index < len(self.labels):
            image = self.images[index]
            label = int(self.labels[index])
        else:
            image, label = self.extra_pairs[index - len(self.labels)]
            label = int(label)
        image = self._to_pil(image)
        return self.transform(image), label


if __name__ == "__main__":
    create_splits(seed=42)
