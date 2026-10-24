"""Classifier training and evaluation entry points."""

"""Train the E0/E2/E3/E4 ResNet-18 classifier from a YAML config."""
import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from src.data import HAMClassifierDataset, CLASS_NAMES
from src.metrics import predict_logits, calculate_metrics, save_confusion_matrix
from src.models import build_classifier
from src.utils import (
    get_device, load_config, make_generator, save_run_config, seed_worker, set_seed,
)

METRIC_COLUMNS = [
    "experiment", "seed", "macro_f1", "accuracy", "bal_acc", "auc",
    "recall_akiec", "recall_bcc", "recall_bkl", "recall_df",
    "recall_mel", "recall_nv", "recall_vasc",
]


def append_metrics(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})


def _load_extra_pairs(cfg):
    """Load optional synthetic images from an NPZ containing images and labels."""
    synthetic = cfg.get("synthetic", {})
    if not synthetic.get("enabled", False):
        return []
    path = synthetic.get("synthetic_path")
    if not path:
        raise ValueError("synthetic.enabled is true but synthetic_path is not set.")
    path = Path(path)
    if not path.is_absolute():
        path = Path(cfg["paths"]["cache_dir"]).parent / path
    if not path.exists():
        raise FileNotFoundError(f"Synthetic image archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if not {"images", "labels"}.issubset(archive.files):
            raise ValueError("Synthetic NPZ must contain 'images' and 'labels' arrays.")
        images, labels = archive["images"], archive["labels"]
    if len(images) != len(labels):
        raise ValueError("Synthetic images and labels lengths differ.")
    return [(images[i], int(labels[i])) for i in range(len(labels))]


def _build_classical_topup_pairs(cfg, base_train_ds):
    """Create class-specific E2 augmentation extras from TRAIN-split images only."""
    topup_cfg = cfg.get("topup", {})
    aug_cfg = cfg.get("augmentation", {})
    if not topup_cfg.get("enabled", False):
        return []
    if aug_cfg.get("mode", "none") != "classical":
        raise ValueError("topup.enabled requires augmentation.mode='classical'.")
    if aug_cfg.get("apply_to", "topup_classes") != "topup_classes":
        raise ValueError("E2 supports augmentation.apply_to='topup_classes' only.")

    classes = topup_cfg.get("classes", [])
    reference_class = topup_cfg.get("reference_class")
    if not classes or reference_class not in CLASS_NAMES:
        raise ValueError("topup.classes must be nonempty and reference_class must be a known class.")
    if any(name not in CLASS_NAMES for name in classes):
        raise ValueError(f"Unknown class in topup.classes: {classes}")

    labels = np.asarray(base_train_ds.labels, dtype=np.int64)
    class_to_id = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    reference_count = int(np.sum(labels == class_to_id[reference_class]))
    target_count = int(round(reference_count * float(topup_cfg.get("target_fraction", 1.0))))
    if target_count < 1:
        raise ValueError(f"Invalid top-up target count: {target_count}")

    color_cfg = aug_cfg.get("color_jitter", {})
    image_size = int(cfg["data"]["image_size"])
    ops = [transforms.Resize((image_size, image_size))]
    if aug_cfg.get("hflip", False):
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if aug_cfg.get("vflip", False):
        ops.append(transforms.RandomVerticalFlip(p=0.5))
    rotation = float(aug_cfg.get("rotation_degrees", 0))
    if rotation:
        ops.append(transforms.RandomRotation(degrees=rotation))
    if any(float(color_cfg.get(k, 0)) for k in ("brightness", "contrast", "saturation", "hue")):
        ops.append(transforms.ColorJitter(
            brightness=float(color_cfg.get("brightness", 0)),
            contrast=float(color_cfg.get("contrast", 0)),
            saturation=float(color_cfg.get("saturation", 0)),
            hue=float(color_cfg.get("hue", 0)),
        ))
    augment = transforms.Compose(ops)

    extras = []
    print(f"Classical top-up target: {target_count} images per configured class "
          f"(reference={reference_class}, count={reference_count}).")
    for class_name in classes:
        class_id = class_to_id[class_name]
        source_indices = np.flatnonzero(labels == class_id)
        current_count = len(source_indices)
        needed = max(0, target_count - current_count)
        print(f"  {class_name}: real={current_count}, adding={needed}, final={current_count + needed}")
        if needed == 0:
            continue
        if current_count == 0:
            raise ValueError(f"Cannot top up {class_name}: no real training images.")
        for _ in range(needed):
            source_idx = int(np.random.choice(source_indices))
            source_image = base_train_ds._to_pil(base_train_ds.images[source_idx])
            augmented = augment(source_image)
            extras.append((np.asarray(augmented, dtype=np.uint8), class_id))

    generated_counts = {name: 0 for name in classes}
    for _, label in extras:
        generated_counts[CLASS_NAMES[int(label)]] += 1
    for class_name in classes:
        expected = max(0, target_count - int(np.sum(labels == class_to_id[class_name])))
        if generated_counts[class_name] != expected:
            raise RuntimeError(f"Top-up count mismatch for {class_name}: expected {expected}, generated {generated_counts[class_name]}")
    return extras


def run_experiment(config_path, smoke=False, max_train_items=256, epochs_override=None):
    cfg = load_config(config_path)
    device = get_device()
    exp_id = cfg["experiment"]["id"]
    seeds = cfg.get("seeds", [0, 1, 2])
    if smoke:
        seeds = [seeds[0]]
    clf_cfg = cfg["classifier"]
    data_cfg = cfg["data"]
    train_cfg = cfg.get("augmentation", {}).get("mode", "none")
    if train_cfg not in {"none", "classical"}:
        raise ValueError(f"Unsupported augmentation mode: {train_cfg}")

    # Keep smoke-test artifacts completely separate from formal experiment outputs.
    if smoke:
        cfg["paths"]["results_dir"] = str(Path(cfg["paths"]["results_dir"]) / "smoke")
        cfg["paths"]["checkpoint_dir"] = str(Path(cfg["paths"]["checkpoint_dir"]) / "smoke")

    cache_path = cfg["paths"]["images_cache"]
    split_path = cfg["paths"]["splits_csv"]
    extra_pairs = _load_extra_pairs(cfg)
    metrics_path = Path(cfg["paths"]["results_dir"]) / "metrics.csv"

    # Formal-run safety check: refuse to overwrite artifacts or duplicate metrics.
    if not smoke:
        existing_run_dirs = [
            Path(cfg["paths"]["results_dir"]) / exp_id / f"seed_{int(seed)}"
            for seed in seeds
        ]
        existing_run_dirs = [path for path in existing_run_dirs if path.exists()]
        if existing_run_dirs:
            raise FileExistsError(
                "Formal run blocked to protect existing artifacts. Existing run "
                f"directories: {existing_run_dirs}. Archive them or use a new experiment ID."
            )
        if metrics_path.exists() and metrics_path.stat().st_size:
            import pandas as pd
            recorded = pd.read_csv(metrics_path)
            if "experiment" in recorded.columns and (recorded["experiment"] == exp_id).any():
                raise FileExistsError(
                    f"Formal run blocked: {exp_id} already has rows in {metrics_path}. "
                    "Reconcile existing results before rerunning under this ID."
                )

    print(f"Experiment: {exp_id}; device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available; training will use CPU.")

    for seed in seeds:
        set_seed(int(seed))
        run_dir = save_run_config(cfg, int(seed))

        do_classical_topup = (
            cfg.get("topup", {}).get("enabled", False)
            and train_cfg == "classical"
            and cfg.get("augmentation", {}).get("apply_to") == "topup_classes"
        )
        if (cfg.get("topup", {}).get("enabled", False)
                and train_cfg == "classical" and not do_classical_topup):
            raise ValueError("Classical top-up requires augmentation.apply_to='topup_classes'.")

        seed_extra_pairs = list(extra_pairs)
        if do_classical_topup:
            # E2: create augmented extras from real TRAIN images only.
            base_train_ds = HAMClassifierDataset(
                cache_path=cache_path, split_csv=split_path, split="train",
                image_size=int(data_cfg["clf_input_size"]), augmentation="none",
            )
            seed_extra_pairs.extend(_build_classical_topup_pairs(cfg, base_train_ds))

        # Only E2's added top-up examples are augmented. E3/E4 use their
        # configured GAN NPZ extras and leave the original real images unchanged.
        dataset_augmentation = "none" if do_classical_topup else train_cfg
        train_ds = HAMClassifierDataset(
            cache_path=cache_path, split_csv=split_path, split="train",
            image_size=int(data_cfg["clf_input_size"]),
            augmentation=dataset_augmentation, extra_pairs=seed_extra_pairs,
            max_items=max_train_items if smoke else None,
        )
        val_ds = HAMClassifierDataset(
            cache_path=cache_path, split_csv=split_path, split="val",
            image_size=int(data_cfg["clf_input_size"]),
            max_items=128 if smoke else None,
        )
        test_ds = HAMClassifierDataset(
            cache_path=cache_path, split_csv=split_path, split="test",
            image_size=int(data_cfg["clf_input_size"]),
            max_items=128 if smoke else None,
        )

        workers = 0 if smoke else int(clf_cfg.get("num_workers", 2))
        generator = make_generator(int(seed))
        train_loader = DataLoader(
            train_ds, batch_size=int(clf_cfg["batch_size"]), shuffle=True,
            num_workers=workers, worker_init_fn=seed_worker, generator=generator,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=int(clf_cfg["batch_size"]), shuffle=False,
            num_workers=workers, pin_memory=(device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_ds, batch_size=int(clf_cfg["batch_size"]), shuffle=False,
            num_workers=workers, pin_memory=(device.type == "cuda"),
        )

        model = build_classifier(
            num_classes=len(CLASS_NAMES),
            pretrained=bool(clf_cfg.get("pretrained_weights") == "IMAGENET1K_V1"),
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(clf_cfg["lr"]),
            weight_decay=float(clf_cfg.get("weight_decay", 0.0)),
        )
        max_epochs = int(epochs_override or clf_cfg["max_epochs"])
        patience = int(clf_cfg["early_stopping"]["patience"])
        best_f1, best_epoch, stale_epochs = -1.0, 0, 0
        best_path = run_dir / "best_model.pt"
        log_path = run_dir / "training_log.csv"

        with log_path.open("w", newline="", encoding="utf-8") as f:
            log_writer = csv.DictWriter(
                f, fieldnames=["epoch", "train_loss", "val_macro_f1", "val_accuracy"]
            )
            log_writer.writeheader()
            for epoch in range(1, max_epochs + 1):
                model.train()
                losses = []
                for images, labels in train_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))

                val_logits, val_labels = predict_logits(model, val_loader, device)
                val_metrics, _, _ = calculate_metrics(val_labels, val_logits)
                mean_loss = float(np.mean(losses)) if losses else float("nan")
                log_writer.writerow({
                    "epoch": epoch, "train_loss": mean_loss,
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_accuracy": val_metrics["accuracy"],
                })
                f.flush()
                print(
                    f"[{exp_id} seed={seed}] epoch {epoch}/{max_epochs} "
                    f"loss={mean_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )

                if val_metrics["macro_f1"] > best_f1:
                    best_f1 = val_metrics["macro_f1"]
                    best_epoch = epoch
                    stale_epochs = 0
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "experiment": exp_id, "seed": int(seed),
                        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                        "class_names": CLASS_NAMES,
                    }, best_path)
                else:
                    stale_epochs += 1
                    if stale_epochs >= patience:
                        print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}.")
                        break

        # Test only after the best checkpoint has been selected on validation.
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_logits, test_labels = predict_logits(model, test_loader, device)
        test_metrics, cm, _ = calculate_metrics(test_labels, test_logits)
        save_confusion_matrix(cm, run_dir, prefix="test")

        row = {"experiment": exp_id, "seed": int(seed), **test_metrics}
        append_metrics(metrics_path, row)
        (run_dir / "test_metrics.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        print(
            f"[{exp_id} seed={seed}] best_epoch={best_epoch} "
            f"best_val_macro_f1={best_f1:.4f} "
            f"TEST macro_f1={test_metrics['macro_f1']:.4f} "
            f"accuracy={test_metrics['accuracy']:.4f} "
            f"balanced_accuracy={test_metrics['bal_acc']:.4f} "
            f"macro_ovr_auc={test_metrics['auc']:.4f}"
        )

    if not smoke:
        print(f"Per-seed metrics appended to: {metrics_path}")
        _print_seed_summary(metrics_path, exp_id)


def _print_seed_summary(metrics_path, exp_id):
    import pandas as pd
    df = pd.read_csv(metrics_path)
    rows = df[df["experiment"] == exp_id]
    if rows.empty:
        return
    cols = ["macro_f1", "accuracy", "bal_acc", "auc"]
    print(f"\n{exp_id} summary across recorded seeds:")
    print(rows[cols].agg(["mean", "std"]).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e0.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="Run one seed, a small subset, and one epoch to check the classifier chain.")
    parser.add_argument("--max-train-items", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    run_experiment(
        args.config, smoke=args.smoke,
        max_train_items=args.max_train_items, epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
