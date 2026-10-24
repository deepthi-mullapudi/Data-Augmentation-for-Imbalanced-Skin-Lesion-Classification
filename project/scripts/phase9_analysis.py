"""
Phase 9 analysis for the Imbalanced Skin Lesion Classification project.

Run from the project root:
    python .\scripts\phase9_analysis.py

Reads existing result artifacts only. Does not train models or overwrite original results.
Outputs are written to results/phase9_analysis/.
Dependencies: pandas, numpy, matplotlib.
"""
from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
RESULTS = ROOT / "results"
OUT = RESULTS / "phase9_analysis"
EXPERIMENTS = ["E0", "E2", "E3", "E4"]
SEEDS = [0, 1, 2]
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
METRICS_PATH = RESULTS / "metrics.csv"


def ensure_project_root():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {METRICS_PATH}. Run this script from the project root."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)


def save_csv(df, path, index=False):
    df.to_csv(path, index=index)
    print(f"Saved: {path.relative_to(ROOT)}")


def load_headline_metrics():
    df = pd.read_csv(METRICS_PATH)
    df = df[df["experiment"].isin(EXPERIMENTS)].copy()
    if df.empty:
        raise ValueError(f"No E0/E2/E3/E4 rows found in {METRICS_PATH}")
    metric_cols = [
        c for c in ["macro_f1", "accuracy", "bal_acc", "auc"] if c in df.columns
    ]
    summary = df.groupby("experiment")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reindex(EXPERIMENTS).reset_index()
    save_csv(summary, OUT / "tables" / "headline_metrics_mean_sd.csv")
    return df, summary


def load_confusion(experiment, seed):
    path = RESULTS / experiment / f"seed_{seed}" / "test_confusion_matrix.csv"
    if not path.exists():
        print(f"WARNING: missing confusion matrix: {path}")
        return None
    df = pd.read_csv(path, index_col=0)
    # CSV format is true\\pred, with actual classes in rows and predicted classes in columns.
    missing_rows = [c for c in CLASSES if c not in df.index]
    missing_cols = [c for c in CLASSES if c not in df.columns]
    if missing_rows or missing_cols:
        raise ValueError(
            f"Unexpected classes in {path}; missing rows={missing_rows}, columns={missing_cols}"
        )
    return df.loc[CLASSES, CLASSES].astype(int)


def confusion_analysis():
    rows = []
    for exp in EXPERIMENTS:
        matrices = []
        for seed in SEEDS:
            cm = load_confusion(exp, seed)
            if cm is not None:
                matrices.append(cm)
                recall = np.diag(cm.to_numpy()) / cm.sum(axis=1).to_numpy()
                for cls, value in zip(CLASSES, recall):
                    rows.append({"experiment": exp, "seed": seed, "class": cls, "recall": value})
        if not matrices:
            continue
        total = sum(matrices)
        total.index.name = "true\\pred"
        save_csv(total.reset_index(), OUT / "tables" / f"{exp}_confusion_counts_sum.csv")
        normalized = total.div(total.sum(axis=1).replace(0, np.nan), axis=0)
        save_csv(
            normalized.reset_index(),
            OUT / "tables" / f"{exp}_confusion_row_normalized.csv"
        )

        # Plot row-normalized aggregate confusion matrix.
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(normalized.to_numpy(), interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"{exp}: row-normalized test confusion matrix (all available seeds)")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                val = normalized.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion within true class")
        fig.tight_layout()
        path = OUT / "plots" / f"{exp}_confusion_row_normalized.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"Saved: {path.relative_to(ROOT)}")

    recalls = pd.DataFrame(rows)
    if not recalls.empty:
        save_csv(recalls, OUT / "tables" / "per_seed_class_recall.csv")
        mean_recall = recalls.groupby(["experiment", "class"])["recall"].agg(["mean", "std"]).reset_index()
        save_csv(mean_recall, OUT / "tables" / "per_class_recall_mean_sd.csv")

        pivot = mean_recall.pivot(index="class", columns="experiment", values="mean").reindex(CLASSES)
        pivot = pivot.reindex(columns=EXPERIMENTS)
        save_csv(pivot.reset_index(), OUT / "tables" / "per_class_recall_wide.csv")

        x = np.arange(len(CLASSES))
        width = 0.19
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, exp in enumerate(EXPERIMENTS):
            vals = [pivot.loc[c, exp] if exp in pivot.columns else np.nan for c in CLASSES]
            ax.bar(x + (i - 1.5) * width, vals, width, label=exp)
        ax.set_xticks(x, CLASSES)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recall")
        ax.set_title("Per-class test recall (mean across available seeds)")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = OUT / "plots" / "per_class_recall.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print(f"Saved: {path.relative_to(ROOT)}")

        # MEL/NV confusions and rare-class recall side-by-side from summed confusion matrices.
        side_rows = []
        for exp in EXPERIMENTS:
            cm = sum(
                m for m in (load_confusion(exp, seed) for seed in SEEDS) if m is not None
            )
            if cm is None or len(cm) == 0:
                continue
            total = cm.to_numpy()
            idx = {c: i for i, c in enumerate(CLASSES)}
            mel_to_nv = total[idx["mel"], idx["nv"]] / total[idx["mel"], :].sum()
            nv_to_mel = total[idx["nv"], idx["mel"]] / total[idx["nv"], :].sum()
            side_rows.append({
                "experiment": exp,
                "mel_to_nv_fraction": mel_to_nv,
                "nv_to_mel_fraction": nv_to_mel,
                "akiec_recall": total[idx["akiec"], idx["akiec"]] / total[idx["akiec"], :].sum(),
                "df_recall": total[idx["df"], idx["df"]] / total[idx["df"], :].sum(),
                "vasc_recall": total[idx["vasc"], idx["vasc"]] / total[idx["vasc"], :].sum(),
                "mel_recall": total[idx["mel"], idx["mel"]] / total[idx["mel"], :].sum(),
                "nv_recall": total[idx["nv"], idx["nv"]] / total[idx["nv"], :].sum(),
            })
        save_csv(pd.DataFrame(side_rows), OUT / "tables" / "mel_nv_and_minority_analysis.csv")


def training_curve_analysis():
    for metric, ylabel, title, filename in [
        ("train_loss", "Training loss", "Training loss by epoch", "training_loss_curves.png"),
        ("val_macro_f1", "Validation macro-F1", "Validation macro-F1 by epoch", "validation_macro_f1_curves.png"),
        ("val_accuracy", "Validation accuracy", "Validation accuracy by epoch", "validation_accuracy_curves.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = 0
        for exp in EXPERIMENTS:
            for seed in SEEDS:
                path = RESULTS / exp / f"seed_{seed}" / "training_log.csv"
                if not path.exists():
                    print(f"WARNING: missing training log: {path}")
                    continue
                log = pd.read_csv(path)
                if not {"epoch", metric}.issubset(log.columns):
                    print(f"WARNING: expected columns missing in {path}")
                    continue
                ax.plot(log["epoch"], log[metric], alpha=0.65, linewidth=1.4,
                        label=f"{exp} seed {seed}")
                plotted += 1
        if plotted:
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            outpath = OUT / "plots" / filename
            fig.savefig(outpath, dpi=180)
            print(f"Saved: {outpath.relative_to(ROOT)}")
        plt.close(fig)


def fid_inventory():
    # Inventory candidate FID artifacts without assuming a particular schema.
    roots = [
        RESULTS / "fid_evaluation",
        RESULTS / "fid_evaluation_active",
        RESULTS / "fid_evaluation_dcgan_aug",
    ]
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".csv", ".txt"}:
                try:
                    if path.suffix.lower() == ".json":
                        obj = json.loads(path.read_text(encoding="utf-8"))
                        text = json.dumps(obj)
                    else:
                        text = path.read_text(encoding="utf-8", errors="ignore")
                    if re.search(r"\bfid\b|frechet", text, flags=re.I):
                        rows.append({
                            "file": str(path.relative_to(ROOT)),
                            "excerpt": text[:1200].replace("\n", " ")[:1200]
                        })
                except Exception as exc:
                    print(f"WARNING: could not inspect {path}: {exc}")
    if rows:
        save_csv(pd.DataFrame(rows), OUT / "tables" / "fid_artifact_inventory.csv")
    else:
        print("No readable FID-containing JSON/CSV/TXT artifacts found in the expected FID folders.")
        print("Use the previously recorded FID values in the written report if verified there.")


def write_readme(summary):
    lines = [
        "# Phase 9 Analysis",
        "",
        "Generated from existing E0/E2/E3/E4 artifacts. No model training was performed.",
        "",
        "## Headline metrics",
        "",
        "| Experiment | Macro-F1 mean ± SD | Accuracy mean ± SD | Balanced accuracy mean ± SD | AUC mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        def fmt(metric):
            mean = r.get(f"{metric}_mean", np.nan)
            sd = r.get(f"{metric}_std", np.nan)
            if pd.isna(mean):
                return "N/A"
            return f"{mean:.4f} ± {sd:.4f}" if pd.notna(sd) else f"{mean:.4f} ± N/A"
        lines.append(
            f"| {r['experiment']} | {fmt('macro_f1')} | {fmt('accuracy')} | {fmt('bal_acc')} | {fmt('auc')} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Values are mean and sample standard deviation across the available seeds; they are not confidence intervals.",
        "- Confusion matrices are summed across available seeds. Since each seed evaluates the same test set, the aggregate represents repeated model predictions, not additional independent patients.",
        "- Per-class recall is diagonal count divided by the corresponding true-class row total.",
        "- This script does not perform a paired statistical test or bootstrap confidence interval.",
        "- A lower FID does not by itself establish better medical realism or better downstream classifier performance.",
        "- The project heuristic previously specified compares mean macro-F1 gap with the baseline seed SD; this is not a formal significance test.",
        "",
        "## Outputs",
        "",
        "- `tables/headline_metrics_mean_sd.csv`",
        "- `tables/per_seed_class_recall.csv`",
        "- `tables/per_class_recall_mean_sd.csv`",
        "- `tables/per_class_recall_wide.csv`",
        "- `tables/*_confusion_counts_sum.csv`",
        "- `tables/*_confusion_row_normalized.csv`",
        "- `tables/mel_nv_and_minority_analysis.csv`",
        "- `plots/per_class_recall.png`",
        "- `plots/*_confusion_row_normalized.png`",
        "- `plots/training_loss_curves.png`",
        "- `plots/validation_macro_f1_curves.png`",
        "- `plots/validation_accuracy_curves.png`",
        "- `tables/fid_artifact_inventory.csv` if matching FID artifacts are found",
        "",
    ]
    path = OUT / "README_phase9.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {path.relative_to(ROOT)}")


def main():
    ensure_project_root()
    print(f"Project root: {ROOT}")
    print(f"Output folder: {OUT}")
    metric_rows, summary = load_headline_metrics()
    confusion_analysis()
    training_curve_analysis()
    fid_inventory()
    write_readme(summary)
    print("\nPhase 9 artifact generation finished.")
    print(f"Review outputs under: {OUT}")


if __name__ == "__main__":
    main()
