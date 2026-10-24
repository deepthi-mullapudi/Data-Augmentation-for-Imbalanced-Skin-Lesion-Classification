
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.models import ConditionalGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = {
    "dcgan_aug_epoch_150": (
       "checkpoints/GAN_DCGAN_AUG/seed_0/dcgan_epoch_150.pt",
    ),
    "dcgan_aug_epoch_200": (
        "checkpoints/GAN_DCGAN_AUG/seed_0/dcgan_epoch_200.pt",
    ),
    "dcgan_epoch_150": (
        "checkpoints/GAN_DCGAN/seed_0/dcgan_epoch_150.pt",
    ),
    "dcgan_epoch_200": (
        "checkpoints/GAN_DCGAN/seed_0/dcgan_epoch_200.pt",
    ),
    "wgangp_epoch_150": (
        "checkpoints/GAN_WGANGP/seed_0/wgangp_epoch_150.pt",
    ),
    "wgangp_epoch_200": (
        "checkpoints/GAN_WGANGP/seed_0/wgangp_epoch_200.pt",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate class-conditional GAN evaluation images."
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=1000,
        help="Number of images to generate per class per checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Generation batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Base random seed.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/fid_evaluation",
        help="Output directory relative to the project root unless absolute.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=list(CANDIDATES),
        default=list(CANDIDATES),
        help="Checkpoint candidates to evaluate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into existing candidate directories.",
    )
    return parser.parse_args()


def make_generator(checkpoint, device):
    cfg = checkpoint["config"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    classes = data_cfg["classes"]
    num_classes = len(classes)

    generator = ConditionalGenerator(
        num_classes=num_classes,
        z_dim=int(model_cfg["z_dim"]),
        class_embed_dim=int(model_cfg["class_embed_dim"]),
        base_channels=int(model_cfg["base_channels"]),
        image_channels=3,
        init_std=float(model_cfg.get("init_std", 0.02)),
    ).to(device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    return generator, classes, model_cfg, data_cfg


def generate_candidate(
    candidate_name,
    checkpoint_path,
    output_root,
    images_per_class,
    batch_size,
    seed,
    device,
    overwrite,
):
    checkpoint_file = PROJECT_ROOT / checkpoint_path
    if not checkpoint_file.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_file}"
        )

    # These are locally saved project checkpoints.
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    if int(checkpoint["epoch"]) not in (150, 200):
        raise ValueError(
            f"Unexpected checkpoint epoch: {checkpoint['epoch']}"
        )

    generator, classes, model_cfg, data_cfg = make_generator(
        checkpoint, device
    )

    expected_type = (
    "dcgan" if candidate_name.startswith("dcgan_aug_")
    else candidate_name.split("_epoch_")[0]
)
    if checkpoint["gan_type"].lower() != expected_type:
        raise ValueError(
            f"Checkpoint type mismatch for {candidate_name}: "
            f"{checkpoint['gan_type']}"
        )

    candidate_dir = output_root / candidate_name

    if candidate_dir.exists() and any(candidate_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output already contains files: {candidate_dir}\n"
                "Choose a new --output-root or pass --overwrite "
                "if you intentionally want to replace this candidate."
            )

    candidate_dir.mkdir(parents=True, exist_ok=True)

    image_counts = {}
    all_images = []
    all_labels = []

    for class_id, class_name in enumerate(classes):
        class_dir = candidate_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        existing_images = list(class_dir.glob("*.png"))
        if existing_images and not overwrite:
            raise FileExistsError(
                f"Class directory already contains PNG files: {class_dir}"
            )

        if overwrite:
            for old_file in existing_images:
                old_file.unlink()

        class_seed = seed + class_id
        random.seed(class_seed)
        np.random.seed(class_seed)
        torch.manual_seed(class_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(class_seed)

        generated = 0

        with torch.inference_mode():
            for start in tqdm(
                range(0, images_per_class, batch_size),
                desc=f"{candidate_name} / {class_name}",
                leave=False,
            ):
                count = min(batch_size, images_per_class - start)

                noise = torch.randn(
                    count,
                    int(model_cfg["z_dim"]),
                    device=device,
                )
                labels = torch.full(
                    (count,),
                    class_id,
                    dtype=torch.long,
                    device=device,
                )

                # Generator outputs are in [-1, 1].
                fake = generator(noise, labels)
                if fake.ndim != 4 or tuple(fake.shape[1:]) != (3, 64, 64):
                    raise ValueError(
                        f"Unexpected output shape: {tuple(fake.shape)}"
                    )

                # Convert [-1, 1] to uint8 RGB for PNG storage.
                fake_uint8 = (
                    (fake.clamp(-1, 1) + 1.0) * 127.5
                ).round().byte().cpu()

                for offset, image_tensor in enumerate(fake_uint8):
                    image_index = start + offset
                    image_array = (
                        image_tensor.permute(1, 2, 0).numpy()
                    )
                    Image.fromarray(image_array, mode="RGB").save(
                        class_dir / f"{image_index:05d}.png"
                    )

                # Keep arrays in memory only for this smoke/full run.
                # Full evaluation can consume PNG directories directly.
                all_images.append(
                    fake_uint8.permute(0, 2, 3, 1).numpy()
                )
                all_labels.append(
                    np.full(count, class_id, dtype=np.int64)
                )
                generated += count

        image_counts[class_name] = generated

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    npz_path = candidate_dir / "synthetic_images.npz"
    np.savez_compressed(
        npz_path,
        images=images,
        labels=labels,
        class_names=np.asarray(classes),
    )

    metadata = {
        "candidate": candidate_name,
        "checkpoint": str(checkpoint_file.relative_to(PROJECT_ROOT)),
        "epoch": int(checkpoint["epoch"]),
        "gan_type": checkpoint["gan_type"],
        "seed": seed,
        "images_per_class": images_per_class,
        "class_names": classes,
        "class_counts": image_counts,
        "image_shape": [64, 64, 3],
        "pixel_format": "uint8 RGB, range 0-255",
        "generator_output_range": data_cfg.get("pixel_range", [-1, 1]),
        "model_config": model_cfg,
        "data_config": data_cfg,
    }

    with open(candidate_dir / "generation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCompleted: {candidate_name}")
    print(f"Output directory: {candidate_dir}")
    print(f"NPZ: {npz_path}")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Class counts: {image_counts}")


def main():
    args = parse_args()

    if args.images_per_class < 1 or args.batch_size < 1:
        raise ValueError(
            "--images-per-class and --batch-size must be positive."
        )

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Images per class: {args.images_per_class}")
    print(f"Output root: {output_root}")

    for candidate_name in args.candidates:
        checkpoint_path = CANDIDATES[candidate_name][0]
        generate_candidate(
            candidate_name=candidate_name,
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            images_per_class=args.images_per_class,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
