from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "dcgan": ROOT / "results" / "fid_evaluation_dcgan_aug"
              / "dcgan_aug_epoch_200" / "synthetic_images.npz",
    "wgangp": ROOT / "results" / "fid_evaluation"
               / "wgangp_epoch_200" / "synthetic_images.npz",
}

OUTPUTS = {
    "dcgan": ROOT / "cache" / "synthetic_dcgan.npz",
    "wgangp": ROOT / "cache" / "synthetic_wgangp.npz",
}

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
TARGET_COUNTS = {0: 131, 3: 279, 6: 259}
SEED = 42
GRID_DIR = ROOT / "results" / "final_synthetic_sets"

def main():
    # Safety checks: validate all inputs and outputs before writing anything.
    for name, path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} source archive not found: {path}")

    existing = [str(p) for p in OUTPUTS.values() if p.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s):\n" + "\n".join(existing)
        )

    loaded = {}
    for name, path in SOURCES.items():
        with np.load(path, allow_pickle=False) as archive:
            images = archive["images"]
            labels = archive["labels"]
            names = archive["class_names"].tolist()

        if images.shape != (7000, 64, 64, 3) or images.dtype != np.uint8:
            raise ValueError(f"Unexpected image array in {name}: "
                             f"{images.shape}, {images.dtype}")
        if labels.shape != (7000,) or labels.dtype.kind not in "iu":
            raise ValueError(f"Unexpected labels in {name}: "
                             f"{labels.shape}, {labels.dtype}")
        if names != CLASS_NAMES:
            raise ValueError(f"Unexpected class order in {name}: {names}")

        counts = np.bincount(labels.astype(np.int64), minlength=7)
        if len(counts) != 7 or not np.all(counts == 1000):
            raise ValueError(f"Expected 1000 images per class in {name}: {counts}")

        loaded[name] = (images, labels.astype(np.int64, copy=False))

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_epoch": 200,
        "selection_seed": SEED,
        "quality_filter": {
            "enabled": False,
            "reason": "E0 classifier filtering was explicitly omitted."
        },
        "target_reference_class": "bcc",
        "target_fraction": 1.0,
        "synthetic_counts": {
            CLASS_NAMES[label]: count
            for label, count in TARGET_COUNTS.items()
        },
        "sets": {}
    }

    for model_index, name in enumerate(("dcgan", "wgangp")):
        images, labels = loaded[name]
        rng = np.random.default_rng(SEED + model_index)
        selected_images = []
        selected_labels = []

        for label, required in TARGET_COUNTS.items():
            indices = np.flatnonzero(labels == label)
            chosen = rng.choice(indices, size=required, replace=False)
            selected_images.append(images[chosen])
            selected_labels.append(
                np.full(required, label, dtype=np.int64)
            )

        final_images = np.concatenate(selected_images, axis=0)
        final_labels = np.concatenate(selected_labels, axis=0)

        # Shuffle the final archive deterministically.
        order = rng.permutation(len(final_labels))
        final_images = np.ascontiguousarray(final_images[order])
        final_labels = final_labels[order]

        if final_images.shape != (669, 64, 64, 3):
            raise AssertionError(f"Unexpected final shape: {final_images.shape}")

        output = OUTPUTS[name]
        np.savez_compressed(
            output,
            images=final_images,
            labels=final_labels,
            class_names=np.asarray(CLASS_NAMES)
        )

        counts = np.bincount(final_labels, minlength=7)
        expected = np.array([131, 0, 0, 279, 0, 0, 259])
        if not np.array_equal(counts, expected):
            raise AssertionError(f"Unexpected final counts for {name}: {counts}")

        # Contact sheet: eight selected examples per generated class.
        cell = 80
        grid = Image.new("RGB", (8 * cell, 3 * cell), "white")
        draw = ImageDraw.Draw(grid)

        for row, label in enumerate((0, 3, 6)):
            class_indices = np.flatnonzero(final_labels == label)[:8]
            draw.text((row * 0 + 4, row * cell + 2),
                      CLASS_NAMES[label], fill="black")
            for col, idx in enumerate(class_indices):
                image = Image.fromarray(final_images[idx]).resize((64, 64))
                grid.paste(image, (col * cell + 8, row * cell + 16))

        grid_path = GRID_DIR / f"{name}_final_grid.png"
        grid.save(grid_path)

        metadata["sets"][name] = {
            "source": str(SOURCES[name].relative_to(ROOT)),
            "output": str(output.relative_to(ROOT)),
            "shape": list(final_images.shape),
            "dtype": str(final_images.dtype),
            "label_dtype": str(final_labels.dtype),
            "class_counts": {
                CLASS_NAMES[i]: int(counts[i])
                for i in range(7)
            },
            "grid": str(grid_path.relative_to(ROOT))
        }

        print(f"\n{name.upper()}")
        print(f"Saved: {output.relative_to(ROOT)}")
        print(f"Images: {final_images.shape}, {final_images.dtype}")
        print(f"Labels: {final_labels.shape}, {final_labels.dtype}")
        print(f"Class counts: {metadata['sets'][name]['class_counts']}")
        print(f"Grid: {grid_path.relative_to(ROOT)}")

    metadata_path = GRID_DIR / "preparation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nMetadata saved: {metadata_path.relative_to(ROOT)}")
    print("\nBoth final synthetic sets created successfully.")

if __name__ == "__main__":
    main()
