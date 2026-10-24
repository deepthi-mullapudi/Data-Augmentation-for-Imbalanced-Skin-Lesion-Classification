# """Shared helpers: seeding, config loading and logging, device, checkpoints.

# Typical use in any script:

#     from src.utils import set_seed, load_config, save_run_config

#     cfg = load_config("configs/e0.yaml")
#     for seed in cfg["seeds"]:
#         set_seed(seed)
#         run_dir = save_run_config(cfg, seed)   # results/E0/seed_0/config.yaml
#         ...
# """
# import json
# import logging
# import os
# import platform
# import random
# import subprocess
# import sys
# import time
# from pathlib import Path

# import numpy as np
# import torch
# import yaml

# # project/ (this file lives in project/src/)
# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# # Optional environment overrides. Not needed on your own machine; on Colab you
# # can point these at Drive without editing any config file, e.g.
# #   os.environ["GANPROJ_CHECKPOINT_DIR"] = "/content/drive/MyDrive/ganproj/checkpoints"
# _PATH_ENV_OVERRIDES = {
#     "data_dir": "GANPROJ_DATA_DIR",
#     "cache_dir": "GANPROJ_CACHE_DIR",
#     "checkpoint_dir": "GANPROJ_CHECKPOINT_DIR",
#     "results_dir": "GANPROJ_RESULTS_DIR",
# }


# # --------------------------------------------------------------------------
# # Reproducibility
# # --------------------------------------------------------------------------
# def set_seed(seed: int) -> None:
#     """Seed Python, NumPy and PyTorch (CPU and all CUDA devices)."""
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # safe no-op when CUDA is unavailable
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False  # benchmark=True would reintroduce nondeterminism


# def seed_worker(worker_id: int) -> None:
#     """Pass as DataLoader(worker_init_fn=seed_worker) so workers are seeded too."""
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


# def make_generator(seed: int) -> torch.Generator:
#     """Pass as DataLoader(generator=make_generator(seed)) for reproducible shuffling."""
#     g = torch.Generator()
#     g.manual_seed(seed)
#     return g


# def get_device() -> torch.device:
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # --------------------------------------------------------------------------
# # Configs
# # --------------------------------------------------------------------------
# def load_config(path) -> dict:
#     """Load a yaml config and turn every entry under `paths` into an absolute path.

#     Relative paths are resolved against the project root, so scripts work no
#     matter which folder you launch them from. Environment overrides (see
#     _PATH_ENV_OVERRIDES) are applied to the four main directories.
#     """
#     path = Path(path)
#     if not path.is_absolute():
#         path = PROJECT_ROOT / path
#     with open(path, "r") as f:
#         cfg = yaml.safe_load(f)

#     for key, value in cfg.get("paths", {}).items():
#         env_name = _PATH_ENV_OVERRIDES.get(key)
#         if env_name and os.environ.get(env_name):
#             value = os.environ[env_name]
#         p = Path(value)
#         cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)

#     cfg["_config_file"] = str(path)
#     return cfg


# def _flatten(d: dict, prefix: str = "") -> dict:
#     out = {}
#     for k, v in d.items():
#         key = f"{prefix}{k}"
#         if isinstance(v, dict):
#             out.update(_flatten(v, prefix=key + "."))
#         else:
#             out[key] = v
#     return out


# # Sections that must be identical in every classifier experiment.
# FIXED_SECTIONS = ("seeds", "split", "data", "classifier")


# def check_fixed_settings(config_paths, fixed_sections=FIXED_SECTIONS) -> None:
#     """Raise if any 'must stay fixed' setting differs between experiment configs.

#     This is the programmatic proof that only the training data changes:
#     call it once at the start of a run, or from a notebook, with the list of
#     configs you are comparing (e.g. e0, e2, e3, e4).
#     """
#     flats = {}
#     for p in config_paths:
#         cfg = load_config(p)
#         flat = _flatten({s: cfg[s] for s in fixed_sections})
#         flats[cfg["experiment"]["id"]] = flat

#     ids = list(flats)
#     ref_id, ref = ids[0], flats[ids[0]]
#     problems = []
#     for other_id in ids[1:]:
#         other = flats[other_id]
#         for key in sorted(set(ref) | set(other)):
#             if ref.get(key) != other.get(key):
#                 problems.append(f"{key}: {ref_id}={ref.get(key)!r} vs {other_id}={other.get(key)!r}")
#     if problems:
#         raise ValueError("Fixed settings differ between experiments:\n  " + "\n  ".join(problems))
#     print(f"OK: fixed settings {list(fixed_sections)} identical across {ids}")


# def _git_commit() -> str:
#     try:
#         return subprocess.check_output(
#             ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
#         ).decode().strip()
#     except Exception:
#         return "unknown"


# def run_dir_for(cfg: dict, seed: int) -> Path:
#     """results/<experiment id>/seed_<n>/  (created if missing)."""
#     d = Path(cfg["paths"]["results_dir"]) / cfg["experiment"]["id"] / f"seed_{seed}"
#     d.mkdir(parents=True, exist_ok=True)
#     return d


# def save_run_config(cfg: dict, seed: int) -> Path:
#     """Write the exact config and run metadata next to the results of this run.

#     Creates in results/<exp>/seed_<n>/:
#       config.yaml   the full config used (paths already resolved)
#       run_meta.json seed, time, git commit, library versions, device
#     Returns the run directory so you can save metrics and figures beside it.
#     """
#     run_dir = run_dir_for(cfg, seed)
#     snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
#     snapshot["seed"] = seed
#     with open(run_dir / "config.yaml", "w") as f:
#         yaml.safe_dump(snapshot, f, sort_keys=False)

#     meta = {
#         "experiment": cfg["experiment"]["id"],
#         "seed": seed,
#         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "config_file": cfg.get("_config_file"),
#         "git_commit": _git_commit(),
#         "python": sys.version.split()[0],
#         "torch": torch.__version__,
#         "cuda": torch.version.cuda,
#         "device": str(get_device()),
#         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
#         "platform": platform.platform(),
#     }
#     with open(run_dir / "run_meta.json", "w") as f:
#         json.dump(meta, f, indent=2)
#     return run_dir


# # --------------------------------------------------------------------------
# # Logging and checkpoints
# # --------------------------------------------------------------------------
# def get_logger(name: str, log_file=None) -> logging.Logger:
#     logger = logging.getLogger(name)
#     if logger.handlers:  # avoid duplicate handlers in notebooks / reruns
#         return logger
#     logger.setLevel(logging.INFO)
#     fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
#     sh = logging.StreamHandler()
#     sh.setFormatter(fmt)
#     logger.addHandler(sh)
#     if log_file is not None:
#         Path(log_file).parent.mkdir(parents=True, exist_ok=True)
#         fh = logging.FileHandler(log_file)
#         fh.setFormatter(fmt)
#         logger.addHandler(fh)
#     return logger


# def save_checkpoint(path, **state) -> None:
#     """save_checkpoint('checkpoints/dcgan_ep050.pt', epoch=50, G=G.state_dict(), D=D.state_dict(),
#     opt_G=optG.state_dict(), opt_D=optD.state_dict())"""
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     tmp = path.with_suffix(path.suffix + ".tmp")
#     torch.save(state, tmp)
#     tmp.replace(path)  # atomic: a crash mid-save never leaves a corrupt checkpoint


# def load_checkpoint(path, map_location=None) -> dict:
#     return torch.load(path, map_location=map_location or get_device())

"""Shared helpers: seeding, config loading and logging, device, checkpoints.

Typical use in any script:

    from src.utils import set_seed, load_config, save_run_config

    cfg = load_config("configs/e0.yaml")
    for seed in cfg["seeds"]:
        set_seed(seed)
        run_dir = save_run_config(cfg, seed)   # results/E0/seed_0/config.yaml
        ...
"""
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# project/ (this file lives in project/src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Optional environment overrides. Not needed on your own machine; on Colab you
# can point these at Drive without editing any config file, e.g.
#   os.environ["GANPROJ_CHECKPOINT_DIR"] = "/content/drive/MyDrive/ganproj/checkpoints"
_PATH_ENV_OVERRIDES = {
    "data_dir": "GANPROJ_DATA_DIR",
    "cache_dir": "GANPROJ_CACHE_DIR",
    "checkpoint_dir": "GANPROJ_CHECKPOINT_DIR",
    "results_dir": "GANPROJ_RESULTS_DIR",
}


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe no-op when CUDA is unavailable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # benchmark=True would reintroduce nondeterminism


def seed_worker(worker_id: int) -> None:
    """Pass as DataLoader(worker_init_fn=seed_worker) so workers are seeded too."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Pass as DataLoader(generator=make_generator(seed)) for reproducible shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------
def load_config(path) -> dict:
    """Load a yaml config and turn every entry under `paths` into an absolute path.

    Relative paths are resolved against the project root, so scripts work no
    matter which folder you launch them from. Environment overrides (see
    _PATH_ENV_OVERRIDES) are applied to the four main directories.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg.get("paths", {}).items():
        env_name = _PATH_ENV_OVERRIDES.get(key)
        if env_name and os.environ.get(env_name):
            value = os.environ[env_name]
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)

    cfg["_config_file"] = str(path)
    return cfg


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key + "."))
        else:
            out[key] = v
    return out


# Sections that must be identical in every classifier experiment.
FIXED_SECTIONS = ("seeds", "split", "data", "classifier")


def check_fixed_settings(config_paths, fixed_sections=FIXED_SECTIONS) -> None:
    """Raise if any 'must stay fixed' setting differs between experiment configs.

    This is the programmatic proof that only the training data changes:
    call it once at the start of a run, or from a notebook, with the list of
    configs you are comparing (e.g. e0, e2, e3, e4).
    """
    flats = {}
    for p in config_paths:
        cfg = load_config(p)
        flat = _flatten({s: cfg[s] for s in fixed_sections})
        flats[cfg["experiment"]["id"]] = flat

    ids = list(flats)
    ref_id, ref = ids[0], flats[ids[0]]
    problems = []
    for other_id in ids[1:]:
        other = flats[other_id]
        for key in sorted(set(ref) | set(other)):
            if ref.get(key) != other.get(key):
                problems.append(f"{key}: {ref_id}={ref.get(key)!r} vs {other_id}={other.get(key)!r}")
    if problems:
        raise ValueError("Fixed settings differ between experiments:\n  " + "\n  ".join(problems))
    print(f"OK: fixed settings {list(fixed_sections)} identical across {ids}")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def run_dir_for(cfg: dict, seed: int) -> Path:
    """results/<experiment id>/seed_<n>/  (created if missing)."""
    d = Path(cfg["paths"]["results_dir"]) / cfg["experiment"]["id"] / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run_config(cfg: dict, seed: int) -> Path:
    """Write the exact config and run metadata next to the results of this run.

    Creates in results/<exp>/seed_<n>/:
      config.yaml   the full config used (paths already resolved)
      run_meta.json seed, time, git commit, library versions, device
    Returns the run directory so you can save metrics and figures beside it.
    """
    run_dir = run_dir_for(cfg, seed)
    snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
    snapshot["seed"] = seed
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)

    meta = {
        "experiment": cfg["experiment"]["id"],
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_file": cfg.get("_config_file"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(get_device()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return run_dir


# --------------------------------------------------------------------------
# Logging and checkpoints
# --------------------------------------------------------------------------
def get_logger(name: str, log_file=None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers in notebooks / reruns
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_checkpoint(path, **state) -> None:
    """save_checkpoint('checkpoints/dcgan_ep050.pt', epoch=50, G=G.state_dict(), D=D.state_dict(),
    opt_G=optG.state_dict(), opt_D=optD.state_dict())"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never leaves a corrupt checkpoint


def load_checkpoint(path, map_location=None) -> dict:
    return torch.load(path, map_location=map_location or get_device())
