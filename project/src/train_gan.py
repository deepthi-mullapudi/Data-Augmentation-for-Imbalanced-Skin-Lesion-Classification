"""DCGAN and WGAN-GP training loops."""

"""Train conditional DCGAN or WGAN-GP on the HAM10000 training split only.

Examples from the project root:
    python -m src.train_gan --config configs/gan_dcgan.yaml --epochs 1 --batch-size 8
    python -m src.train_gan --config configs/gan_dcgan.yaml
    python -m src.train_gan --config configs/gan_wgangp.yaml --epochs 1 --batch-size 8

The periodic checkpoints are candidates, not automatically selected "best" models.
Select a checkpoint later using visual review and Phase 6 FID analysis.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.utils import save_image

from src.models import ConditionalCritic, ConditionalDiscriminator, ConditionalGenerator
from src.utils import get_device, load_config, save_run_config, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


class HANGANTrainingDataset(Dataset):
    """Reads cached uint8 images, selecting only rows assigned to train."""
    def __init__(self, cache_path, split_csv, max_items=None):
        cache_path = Path(cache_path)
        split_csv = Path(split_csv)
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        if not split_csv.is_absolute():
            split_csv = PROJECT_ROOT / split_csv
        if not cache_path.exists():
            raise FileNotFoundError(f"Image cache not found: {cache_path}")
        if not split_csv.exists():
            raise FileNotFoundError(f"Split manifest not found: {split_csv}")

        with np.load(cache_path, allow_pickle=False) as cache:
            images = cache["images"]
            labels = cache["labels"]
            splits = cache["split"].astype(str)
            image_ids = cache["image_ids"].astype(str)

        manifest = pd.read_csv(split_csv)
        required = {"image_id", "dx", "split"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Split manifest missing columns: {sorted(missing)}")
        if len(manifest) != len(images):
            raise ValueError(f"Cache/manifest length mismatch: {len(images)} vs {len(manifest)}")
        if not np.array_equal(image_ids, manifest["image_id"].astype(str).to_numpy()):
            raise ValueError("Cache image_ids do not align with split manifest row order.")
        if not np.array_equal(splits, manifest["split"].astype(str).to_numpy()):
            raise ValueError("Cache split array does not align with split manifest.")
        label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
        expected_labels = manifest["dx"].map(label_map).to_numpy()
        if not np.array_equal(labels, expected_labels):
            raise ValueError("Cache labels do not align with manifest diagnostic labels.")

        train_mask = splits == "train"
        self.images = images[train_mask]
        self.labels = labels[train_mask].astype(np.int64)
        self.image_ids = image_ids[train_mask].tolist()
        if max_items is not None:
            if max_items < 7:
                raise ValueError("--max-train-items must be at least 7 to retain all classes.")
            # Deterministic round-robin subset; keeps all classes represented.
            chosen = []
            class_indices = {c: np.flatnonzero(self.labels == c).tolist() for c in range(7)}
            offsets = {c: 0 for c in range(7)}
            while len(chosen) < max_items:
                progressed = False
                for c in range(7):
                    if offsets[c] < len(class_indices[c]) and len(chosen) < max_items:
                        chosen.append(class_indices[c][offsets[c]])
                        offsets[c] += 1
                        progressed = True
                if not progressed:
                    break
            chosen = np.asarray(chosen, dtype=np.int64)
            self.images = self.images[chosen]
            self.labels = self.labels[chosen]
            self.image_ids = [self.image_ids[i] for i in chosen]

        if len(self.labels) == 0:
            raise ValueError("No training images were selected.")
        if set(np.unique(self.labels).tolist()) != set(range(7)):
            raise ValueError("GAN training subset must contain all seven classes.")
        if self.images.ndim != 4 or self.images.shape[1:] != (64, 64, 3):
            raise ValueError(f"Expected cached images with shape (N,64,64,3), got {self.images.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # uint8 [0,255] -> float32 [-1,1], then HWC -> CHW.
        image = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float()
        image = image.div(127.5).sub(1.0)
        return image, torch.tensor(int(self.labels[index]), dtype=torch.long)


def make_balanced_sampler(labels):
    """GAN-only replacement sampler; each class has equal total sampling weight."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=7)
    if np.any(counts == 0):
        raise ValueError(f"Cannot balance sampler; missing class counts: {counts.tolist()}")
    per_sample_weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(per_sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def diff_augment(images, policy):
    """Lightweight differentiable color, translation and cutout augmentation."""
    policies = {item.strip().lower() for item in str(policy).split(",") if item.strip()}
    x = images
    batch, channels, height, width = x.shape

    if "color" in policies:
        # Per-image brightness, contrast, and saturation factors.
        brightness = torch.rand(batch, 1, 1, 1, device=x.device) - 0.5
        x = x + brightness
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        contrast = 0.5 + torch.rand(batch, 1, 1, 1, device=x.device)
        x = (x - mean) * contrast + mean
        if channels == 3:
            gray = x.mean(dim=1, keepdim=True)
            saturation = 0.5 + torch.rand(batch, 1, 1, 1, device=x.device)
            x = (x - gray) * saturation + gray

    if "translation" in policies:
        # Random integer shifts implemented with grid_sample.
        max_shift_x = max(1, int(width * 0.125))
        max_shift_y = max(1, int(height * 0.125))
        shifts_x = torch.randint(-max_shift_x, max_shift_x + 1, (batch,), device=x.device)
        shifts_y = torch.randint(-max_shift_y, max_shift_y + 1, (batch,), device=x.device)
        theta = torch.zeros(batch, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1
        theta[:, 0, 2] = -2.0 * shifts_x.to(x.dtype) / max(width - 1, 1)
        theta[:, 1, 2] = -2.0 * shifts_y.to(x.dtype) / max(height - 1, 1)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)

    if "cutout" in policies:
        cut_h, cut_w = max(1, height // 4), max(1, width // 4)
        mask = torch.ones(batch, 1, height, width, device=x.device, dtype=x.dtype)
        centers_y = torch.randint(0, height, (batch,), device=x.device)
        centers_x = torch.randint(0, width, (batch,), device=x.device)
        for i in range(batch):
            y0 = max(0, int(centers_y[i]) - cut_h // 2)
            y1 = min(height, y0 + cut_h)
            x0 = max(0, int(centers_x[i]) - cut_w // 2)
            x1 = min(width, x0 + cut_w)
            mask[i, :, y0:y1, x0:x1] = 0
        x = x * mask
    return x.clamp(-1, 1)



def gradient_penalty(critic, real_images, fake_images, labels, device):
    batch_size = real_images.size(0)

    alpha = torch.rand(
        batch_size, 1, 1, 1, device=device
    )

    interpolated = (
        alpha * real_images + (1 - alpha) * fake_images
    ).requires_grad_(True)

    scores = critic(interpolated, labels)

    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    gradient_norms = gradients.norm(2, dim=1)

    penalty = ((gradient_norms - 1) ** 2).mean()
    mean_gradient_norm = gradient_norms.detach().mean().item()

    return penalty, mean_gradient_norm

def save_sample_grid(generator, fixed_noise, fixed_labels, output_path, chunk_size=8):
    """Save fixed-noise samples in chunks to reduce peak GPU memory."""
    was_training = generator.training
    generator.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(fixed_noise), chunk_size):
            noise_chunk = fixed_noise[start:start + chunk_size]
            label_chunk = fixed_labels[start:start + chunk_size]
            chunks.append(generator(noise_chunk, label_chunk).cpu())
        images = torch.cat(chunks, dim=0)
        print(
            "Sample tensor diagnostics:",
            f"shape={tuple(images.shape)}",
            f"min={images.min().item():.4f}",
            f"max={images.max().item():.4f}",
            f"mean={images.mean().item():.4f}",
            f"std={images.std().item():.4f}",
        )
        images = (images.clamp(-1, 1) + 1) / 2
        save_image(images, str(output_path), nrow=8, padding=2)
    if was_training:
        generator.train()


def save_training_checkpoint(path, epoch, generator, discriminator, opt_g, opt_d, cfg, seed, kind):
    state = {
        "epoch": epoch,
        "experiment": cfg["experiment"]["id"],
        "seed": int(seed),
        "gan_type": kind,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": opt_g.state_dict(),
        "optimizer_d_state_dict": opt_d.state_dict(),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "note": "Training checkpoint candidate; not automatically selected as best.",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def run_training(config_path, epochs_override=None, batch_size_override=None, max_train_items=None,
                 smoke=False, num_workers=0):
    cfg = load_config(config_path)
    device = get_device()
    exp = cfg["experiment"]["id"]
    kind = str(cfg["model"]["type"]).lower()
    if kind not in {"dcgan", "wgangp"}:
        raise ValueError(f"Unsupported model.type {kind!r}; expected dcgan or wgangp.")

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    seeds = cfg.get("seeds", [0])
    epochs = int(epochs_override if epochs_override is not None else train_cfg["epochs"])
    batch_size = int(batch_size_override if batch_size_override is not None else train_cfg["batch_size"])
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive.")
    if smoke:
        seeds = [seeds[0]]
        max_train_items = max_train_items or 70
        epochs = epochs_override or 1
        batch_size = batch_size_override or min(batch_size, 8)

    print(f"Experiment: {exp} | type: {kind} | device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {epochs} | batch size: {batch_size} | balanced sampler: {train_cfg.get('balanced_sampler', True)}")

    for seed in seeds:
        set_seed(int(seed))
        run_dir = save_run_config(cfg, int(seed))
        if smoke:
            run_dir = run_dir / "smoke"
            run_dir.mkdir(parents=True, exist_ok=True)

        dataset = HANGANTrainingDataset(
            cfg["paths"]["images_cache"], cfg["paths"]["splits_csv"], max_items=max_train_items
        )
        labels = dataset.labels
        counts = np.bincount(labels, minlength=7)
        print(f"Training images: {len(dataset)} | class counts: {dict(zip(CLASS_NAMES, counts.tolist()))}")
        if train_cfg.get("balanced_sampler", True):
            sampler = make_balanced_sampler(labels)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )
        if len(loader) == 0:
            raise ValueError("DataLoader has zero batches; increase dataset size or reduce batch size.")

        configured_classes = list(data_cfg["classes"])
        if configured_classes != CLASS_NAMES:
            raise ValueError(
                f"Config class order {configured_classes} does not match fixed label order {CLASS_NAMES}."
            )
        num_classes = len(configured_classes)
        z_dim = int(model_cfg["z_dim"])
        embed_dim = int(model_cfg["class_embed_dim"])
        base_channels = int(model_cfg.get("base_channels", 512))
        init_std = float(model_cfg.get("init_std", 0.02))

        generator = ConditionalGenerator(
            num_classes=num_classes, z_dim=z_dim, class_embed_dim=embed_dim,
            base_channels=base_channels, init_std=init_std,
        ).to(device)

        if kind == "dcgan":
            discriminator = ConditionalDiscriminator(
                num_classes=num_classes, class_embed_dim=embed_dim,
                base_channels=64, init_std=init_std,
            ).to(device)
            criterion = nn.BCEWithLogitsLoss()
            betas = tuple(float(v) for v in train_cfg.get("betas", [0.5, 0.999]))
            opt_g = torch.optim.Adam(generator.parameters(), lr=float(train_cfg["lr_g"]), betas=betas)
            opt_d = torch.optim.Adam(discriminator.parameters(), lr=float(train_cfg["lr_d"]), betas=betas)
        else:
            norm = str(model_cfg.get("d_norm", "layernorm")).lower()
            discriminator = ConditionalCritic(
                num_classes=num_classes, class_embed_dim=embed_dim,
                base_channels=64, norm=norm, init_std=init_std,
            ).to(device)
            betas = tuple(float(v) for v in train_cfg.get("betas", [0.0, 0.9]))
            opt_g = torch.optim.Adam(generator.parameters(), lr=float(train_cfg["lr_g"]), betas=betas)
            opt_d = torch.optim.Adam(discriminator.parameters(), lr=float(train_cfg["lr_d"]), betas=betas)

        # One fixed noise/label set makes epoch-to-epoch image grids comparable.
        samples_per_class = int(cfg.get("logging", {}).get("samples_per_class", 8))
        fixed_labels = torch.arange(num_classes, device=device).repeat_interleave(samples_per_class)
        fixed_noise = torch.randn(len(fixed_labels), z_dim, device=device)
        sample_every = max(1, int(cfg.get("logging", {}).get("sample_every", 5)))
        checkpoint_every = max(1, int(cfg.get("logging", {}).get("checkpoint_every", 5)))
        ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / exp / f"seed_{seed}"
        if smoke:
            ckpt_dir = ckpt_dir / "smoke"
        sample_dir = run_dir / "samples"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "training_log.csv"
        
        fields = [
            "epoch",
            "g_loss",
            "d_loss",
            "d_real_mean",
            "d_fake_mean",
            "wasserstein_estimate",
            "gradient_norm_mean",
            "epoch_seconds",
            "images_seen",
            "gan_type",
        ]

        history = []

        with log_path.open("w", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=fields)
            writer.writeheader()
            images_seen = 0
            for epoch in range(1, epochs + 1):
                start = time.perf_counter()
                generator.train()
                discriminator.train()
                g_losses, d_losses, real_scores, fake_scores = [], [], [], []
                gradient_norms = []

                for real_images, real_labels in loader:
                    real_images = real_images.to(device, non_blocking=True)
                    real_labels = real_labels.to(device, non_blocking=True)
                    batch = real_images.size(0)
                    images_seen += batch
                    noise = torch.randn(batch, z_dim, device=device)
                    # For WGAN-GP, use the real labels for generated samples in this
                    # critic step so each interpolation has one consistent condition.
                    # The balanced real sampler therefore also balances fake labels.
                    if kind == "wgangp":
                        fake_labels = real_labels
                    else:
                        fake_labels = torch.randint(0, num_classes, (batch,), device=device)
                    fake_images = generator(noise, fake_labels)

                    aug = cfg.get("train", {}).get("diffaugment", {})
                    use_aug = bool(aug.get("enabled", False))
                    policy = aug.get("policy", "color,translation,cutout")
                    real_for_d = diff_augment(real_images, policy) if use_aug else real_images
                    fake_for_d = diff_augment(fake_images.detach(), policy) if use_aug else fake_images.detach()

                    if kind == "dcgan":
                        opt_d.zero_grad(set_to_none=True)
                        real_logits = discriminator(real_for_d, real_labels)
                        fake_logits = discriminator(fake_for_d, fake_labels)
                        real_targets = torch.full_like(real_logits, float(train_cfg.get("real_label", 0.9)))
                        fake_targets = torch.full_like(fake_logits, float(train_cfg.get("fake_label", 0.0)))
                        d_loss = (criterion(real_logits, real_targets) + criterion(fake_logits, fake_targets)) / 2
                        d_loss.backward()
                        opt_d.step()

                        opt_g.zero_grad(set_to_none=True)
                        noise_g = torch.randn(batch, z_dim, device=device)
                        labels_g = torch.randint(0, num_classes, (batch,), device=device)
                        generated = generator(noise_g, labels_g)
                        generated_for_d = diff_augment(generated, policy) if use_aug else generated
                        gen_logits = discriminator(generated_for_d, labels_g)
                        g_targets = torch.ones_like(gen_logits)
                        g_loss = criterion(gen_logits, g_targets)
                        g_loss.backward()
                        opt_g.step()

                        real_value = torch.sigmoid(real_logits.detach()).mean().item()
                        fake_value = torch.sigmoid(fake_logits.detach()).mean().item()
                    else:
                        opt_d.zero_grad(set_to_none=True)
                        real_scores_batch = discriminator(real_for_d, real_labels)
                        fake_scores_batch = discriminator(fake_for_d, fake_labels)
                        gp, mean_gradient_norm = gradient_penalty(
                            discriminator,
                            real_for_d,
                            fake_for_d,
                            real_labels,
                            device,
                        )
                        gradient_norms.append(mean_gradient_norm)
                        d_loss = fake_scores_batch.mean() - real_scores_batch.mean() + float(
                            train_cfg.get("gp_lambda", 10.0)
                        ) * gp
                        d_loss.backward()
                        opt_d.step()

                        n_critic = max(1, int(train_cfg.get("n_critic", 5)))
                        g_loss = None
                        if (epoch_batch_index := (len(d_losses) + 1)) % n_critic == 0:
                            opt_g.zero_grad(set_to_none=True)
                            noise_g = torch.randn(batch, z_dim, device=device)
                            labels_g = torch.randint(0, num_classes, (batch,), device=device)
                            generated = generator(noise_g, labels_g)
                            generated_for_d = diff_augment(generated, policy) if use_aug else generated
                            g_loss = -discriminator(generated_for_d, labels_g).mean()
                            g_loss.backward()
                            opt_g.step()

                        real_value = real_scores_batch.detach().mean().item()
                        fake_value = fake_scores_batch.detach().mean().item()

                    d_losses.append(float(d_loss.detach().item()))
                    if g_loss is not None:
                        g_losses.append(float(g_loss.detach().item()))
                    real_scores.append(real_value)
                    fake_scores.append(fake_value)

                elapsed = time.perf_counter() - start
                row = {
                    "epoch": epoch,
                    "g_loss": float(np.mean(g_losses)) if g_losses else float("nan"),
                    "d_loss": float(np.mean(d_losses)) if d_losses else float("nan"),
                    "d_real_mean": float(np.mean(real_scores)) if real_scores else float("nan"),
                    "d_fake_mean": float(np.mean(fake_scores)) if fake_scores else float("nan"),
                    "wasserstein_estimate": (
                        float(np.mean(real_scores) - np.mean(fake_scores))
                        if kind == "wgangp"
                        else float("nan")
                    ),
                    "gradient_norm_mean": (
                        float(np.mean(gradient_norms))
                        if gradient_norms
                        else float("nan")
                    ),
                    "epoch_seconds": round(elapsed, 3),
                    "images_seen": images_seen,
                    "gan_type": kind,
                }
                writer.writerow(row)
                log_file.flush()
                history.append(row)
                score_note = "probabilities" if kind == "dcgan" else "raw critic scores"
                
                print(
                    f"[{exp} seed={seed}] epoch {epoch}/{epochs} "
                    f"D={row['d_loss']:.4f} G={row['g_loss']:.4f} "
                    f"D(real)={row['d_real_mean']:.4f} D(fake)={row['d_fake_mean']:.4f} "
                    f"({score_note}) time={elapsed:.1f}s"
                )
                
                if kind == "wgangp":
                    print(
                        f"    Wasserstein estimate="
                        f"{row['wasserstein_estimate']:.4f} "
                        f"Gradient norm="
                        f"{row['gradient_norm_mean']:.4f}"
                    )


                if epoch % sample_every == 0 or epoch == 1 or epoch == epochs:
                    grid_path = sample_dir / f"epoch_{epoch:03d}.png"
                    save_sample_grid(generator, fixed_noise, fixed_labels, grid_path)

                if epoch % checkpoint_every == 0 or epoch == epochs:
                    checkpoint_path = ckpt_dir / f"{kind}_epoch_{epoch:03d}.pt"
                    save_training_checkpoint(
                        checkpoint_path, epoch, generator, discriminator, opt_g, opt_d, cfg, seed, kind
                    )
                    # Always maintain a clearly named latest checkpoint. It is not
                    # labelled as best; best selection is deferred to visual/FID review.
                    latest_path = ckpt_dir / f"{kind}_latest.pt"
                    save_training_checkpoint(
                        latest_path, epoch, generator, discriminator, opt_g, opt_d, cfg, seed, kind
                    )

        summary = {
            "experiment": exp,
            "gan_type": kind,
            "seed": int(seed),
            "epochs_completed": epochs,
            "training_images": len(dataset),
            "training_class_counts": dict(zip(CLASS_NAMES, counts.tolist())),
            "batch_size": batch_size,
            "images_seen": int(images_seen),
            "final_epoch": history[-1] if history else None,
            "checkpoint_directory": str(ckpt_dir),
            "sample_directory": str(sample_dir),
            "checkpoint_selection": "Not automatic; review saved grids and use Phase 6 FID.",
        }
        (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Training log: {log_path}")
        print(f"Samples: {sample_dir}")
        print(f"Checkpoints: {ckpt_dir}")
        print("Run completed. Inspect generated grids before scaling up or selecting a checkpoint.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gan_dcgan.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configured batch size.")
    parser.add_argument("--max-train-items", type=int, default=None,
                        help="Use a deterministic subset for a smoke test; must include all classes.")
    parser.add_argument("--smoke", action="store_true",
                        help="One short epoch on a small, class-inclusive subset; artifacts are isolated.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers; 0 is the safest Windows default.")
    args = parser.parse_args()
    run_training(
        args.config, epochs_override=args.epochs, batch_size_override=args.batch_size,
        max_train_items=args.max_train_items, smoke=args.smoke, num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
