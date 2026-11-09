import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import METRICS_DIR

SEEDS = (42, 52, 62)
VALUE_KEYS = (
    "accuracy",
    "macro_f1",
    "negative_precision",
    "negative_recall",
    "negative_f1",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "train_seconds",
    "peak_vram_mb",
    "model_size_mb",
)


def main() -> None:
    records = []
    for seed in SEEDS:
        path = METRICS_DIR / f"distilbert_seed{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing run metrics: {path}")
        records.append(json.loads(path.read_text(encoding="utf-8")))

    split_hashes = {record.get("split_sha256") for record in records}
    if len(split_hashes) != 1 or None in split_hashes:
        raise ValueError("All runs must record the same shared-split fingerprint")

    rows = []
    for seed, record in zip(SEEDS, records):
        tuned = record["val_metrics"]
        default = record["val_metrics_threshold_05"]
        row = {
            "seed": seed,
            "best_epoch": record["best_epoch"],
            "threshold": record["best_threshold"],
            "accuracy": tuned["accuracy"],
            "macro_f1": tuned["macro_f1"],
            "negative_precision": tuned["negative_precision"],
            "negative_recall": tuned["negative_recall"],
            "negative_f1": tuned["negative_f1"],
            "positive_precision": tuned["positive_precision"],
            "positive_recall": tuned["positive_recall"],
            "positive_f1": tuned["positive_f1"],
            "accuracy_threshold_05": default["accuracy"],
            "macro_f1_threshold_05": default["macro_f1"],
            "train_seconds": record["total_seconds"],
            "peak_vram_mb": record["peak_vram_mb"],
            "model_size_mb": record["model_size_mb"],
        }
        rows.append(row)

    numeric = [key for key in rows[0] if key != "seed" and key != "best_epoch"]
    mean_row = {"seed": "mean", "best_epoch": ""}
    std_row = {"seed": "sample_std", "best_epoch": ""}
    for key in numeric:
        values = [row[key] for row in rows]
        mean_row[key] = float(np.mean(values))
        std_row[key] = float(np.std(values, ddof=1))
    result = pd.DataFrame(rows + [mean_row, std_row])
    result.to_csv(METRICS_DIR / "distilbert_seed_summary.csv", index=False, float_format="%.8f")

    summary = {
        "experiment": "DistilBERT three-seed validation on the shared split",
        "split_sha256": next(iter(split_hashes)),
        "seeds": list(SEEDS),
        "runs": rows,
        "mean": mean_row,
        "sample_std": std_row,
        "interpretation": {
            "val_metrics": "Per-seed threshold selected for validation Macro-F1.",
            "val_metrics_threshold_05": "Common 0.5 threshold for comparison with TF-IDF predictions.",
        },
    }
    (METRICS_DIR / "distilbert_seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (METRICS_DIR / "distilbert_metrics.json").write_text(
        json.dumps(records[-1], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Wrote shared-split three-seed summary and seed 62 compatibility metrics.")


if __name__ == "__main__":
    main()
