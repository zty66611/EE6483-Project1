"""Compute metrics and generate figures for model comparison.

Shared by every model: call ``compute_metrics`` / ``log_run`` so all experiments use one metric
definition and land in ``outputs/metrics/experiments.csv``.

CLI:
    python -m src.evaluate                                 # comparison table + chart from experiments.csv
    python -m src.evaluate --predictions FILE --run-id X   # metrics + confusion matrix for a predictions CSV
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.data import FIGURES_DIR, METRICS_DIR, save_json

EXPERIMENTS_CSV = METRICS_DIR / "experiments.csv"
RUNS_DIR = METRICS_DIR / "runs"
PREDICTIONS_DIR = METRICS_DIR / "predictions"
LABEL_NAMES = ["negative", "positive"]


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, Macro-F1, per-class precision/recall/F1 and the confusion matrix (rows = true)."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
    }
    for i, name in enumerate(LABEL_NAMES):
        metrics[f"precision_{name}"] = float(p[i])
        metrics[f"recall_{name}"] = float(r[i])
        metrics[f"f1_{name}"] = float(f[i])
        metrics[f"support_{name}"] = int(s[i])
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return metrics


def plot_confusion_matrix(cm, title: str, path) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_predictions(run_id: str, df: pd.DataFrame, y_pred, prob_pos=None) -> None:
    """Validation predictions in a common format for case analysis: orig_index, text, label, pred, prob."""
    out = df[["orig_index", "text", "sentiments"]].copy()
    out["pred"] = np.asarray(y_pred).astype(int)
    if prob_pos is not None:
        out["prob_positive"] = np.round(prob_pos, 5)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTIONS_DIR / f"{run_id}_val.csv", index=False)


def log_run(run_id: str, model: str, config: dict, metrics: dict, resources: dict) -> dict:
    """Write the full run record to runs/<run_id>.json, update experiments.csv and draw its confusion matrix."""
    record = {
        "run_id": run_id,
        "model": model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "metrics": metrics,
        "resources": resources,
    }
    save_json(record, RUNS_DIR / f"{run_id}.json")

    row = {"run_id": run_id, "model": model, "timestamp": record["timestamp"],
           "seed": config.get("random_seed")}
    row.update({k: round(v, 4) if isinstance(v, float) else v
                for k, v in metrics.items() if k != "confusion_matrix"})
    row.update({k: round(v, 3) if isinstance(v, float) else v for k, v in resources.items()})
    row["config"] = json.dumps(config, sort_keys=True)
    table = pd.read_csv(EXPERIMENTS_CSV) if EXPERIMENTS_CSV.exists() else pd.DataFrame()
    if not table.empty:
        table = table[table["run_id"] != run_id]  # re-running a run_id replaces its row
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(EXPERIMENTS_CSV, index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics["confusion_matrix"], f"{run_id} (val)", FIGURES_DIR / f"cm_{run_id}.png")
    return record


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal Markdown table writer (avoids the optional tabulate dependency)."""
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def compare(run_ids: list[str] | None = None) -> pd.DataFrame:
    """Markdown comparison table and accuracy / Macro-F1 bar chart for the logged experiments."""
    table = pd.read_csv(EXPERIMENTS_CSV)
    if run_ids:
        table = table[table["run_id"].isin(run_ids)]
    cols = [c for c in ["run_id", "model", "accuracy", "macro_f1", "recall_negative", "recall_positive",
                        "train_time_s", "val_infer_time_s", "peak_ram_mb", "peak_gpu_mem_mb",
                        "model_size_mb"] if c in table]
    view = table[cols]
    (METRICS_DIR / "comparison.md").write_text(to_markdown(view), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(view)), 3.6))
    x = np.arange(len(view))
    ax.bar(x - 0.2, view["accuracy"], 0.4, label="Accuracy", color="#4f81bd")
    ax.bar(x + 0.2, view["macro_f1"], 0.4, label="Macro-F1", color="#f79646")
    ax.set_xticks(x, view["run_id"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Validation performance by experiment")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)
    return view


def main() -> None:
    """Evaluate saved predictions or models."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", help="CSV with columns sentiments and pred")
    parser.add_argument("--run-id", help="name used for the confusion-matrix figure")
    parser.add_argument("--runs", nargs="*", help="restrict the comparison to these run_ids")
    args = parser.parse_args()

    if args.predictions:
        df = pd.read_csv(args.predictions)
        metrics = compute_metrics(df["sentiments"], df["pred"])
        print(json.dumps(metrics, indent=2))
        run_id = args.run_id or "predictions"
        plot_confusion_matrix(metrics["confusion_matrix"], run_id, FIGURES_DIR / f"cm_{run_id}.png")
    else:
        print(to_markdown(compare(args.runs)))


if __name__ == "__main__":
    main()
