"""Controlled-variable tuning of the TF-IDF baseline on the fixed validation split, then 5-fold CV.

Order (one group at a time, the others held fixed; selection metric = validation Macro-F1):
    1. features  word / char / word+char, each with class_weight None / balanced (data is ~85% positive)
    2. max_features and min_df
    3. regularisation C
    4. stratified 5-fold CV of the chosen config on the full deduplicated training data

    python -m src.tune_tfidf --base configs/e2_word_char.json
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import CLEAN_COL, LABEL_COL, METRICS_DIR, ROOT, load_json, load_split, save_json
from src.evaluate import compute_metrics, to_markdown
from src.train_tfidf import build_pipeline, fit_and_evaluate


def set_path(config: dict, path: str, value) -> None:
    keys = path.split(".")
    node = config
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def run_grid(stage: str, base: dict, grid: dict, train, val, rows: list) -> dict:
    """Try every combination in ``grid`` on top of ``base``; return the best config by Macro-F1."""
    best, best_f1 = None, -1.0
    for values in itertools.product(*grid.values()):
        config = copy.deepcopy(base)
        for path, v in zip(grid, values):
            set_path(config, path, v)
        _, m, r, _, _ = fit_and_evaluate(config, train, val)
        rows.append({"stage": stage, **dict(zip(grid, map(str, values))),
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "recall_negative": m["recall_negative"], "recall_positive": m["recall_positive"],
                     "train_time_s": r["train_time_s"], "n_features": r["n_features"]})
        print(f"[{stage}] {dict(zip(grid, values))} macro_f1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")
        if m["macro_f1"] > best_f1:
            best, best_f1 = config, m["macro_f1"]
    print(f"[{stage}] best macro_f1={best_f1:.4f}\n")
    return best


def cross_validate(config: dict, df: pd.DataFrame, folds: int, seed: int) -> dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(df[CLEAN_COL], df[LABEL_COL]):
        pipe = build_pipeline(config).fit(df[CLEAN_COL].iloc[tr], df[LABEL_COL].iloc[tr])
        scores.append(compute_metrics(df[LABEL_COL].iloc[te], pipe.predict(df[CLEAN_COL].iloc[te])))
    out = {"folds": folds}
    for k in ["accuracy", "macro_f1", "recall_negative", "recall_positive"]:
        vals = [s[k] for s in scores]
        out[f"{k}_mean"], out[f"{k}_std"], out[f"{k}_per_fold"] = float(np.mean(vals)), float(np.std(vals)), vals
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="configs/e2_word_char.json")
    parser.add_argument("--out-config", default="configs/tfidf_best.json")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    base = load_json(args.base)
    seed = base.get("random_seed", 42)
    train, val = load_split("train"), load_split("val")
    rows: list[dict] = []

    cfg = run_grid("1_features", base, {
        "features": ["word", "char", "word+char"],
        "classifier.class_weight": [None, "balanced"],
    }, train, val, rows)

    size_grid = {"vectorizer.min_df": [1, 2, 3]}
    if "word" in cfg["features"]:
        size_grid["vectorizer.max_features"] = [30000, 50000, 100000]
    if "char" in cfg["features"]:
        size_grid["char_vectorizer.max_features"] = [30000, 50000, 100000]
    cfg = run_grid("2_size", cfg, size_grid, train, val, rows)

    cfg = run_grid("3_C", cfg, {"classifier.C": [0.1, 0.3, 1, 3, 10]}, train, val, rows)

    cv = cross_validate(cfg, load_split("full"), args.folds, seed)
    print(f"[4_cv] macro_f1={cv['macro_f1_mean']:.4f}±{cv['macro_f1_std']:.4f} "
          f"acc={cv['accuracy_mean']:.4f}±{cv['accuracy_std']:.4f}")

    table = pd.DataFrame(rows)
    table.to_csv(METRICS_DIR / "tfidf_tuning.csv", index=False)
    (METRICS_DIR / "tfidf_tuning.md").write_text(to_markdown(table.fillna("")), encoding="utf-8")
    save_json(cfg, ROOT / args.out_config)
    save_json({"config": cfg, "cv": cv}, METRICS_DIR / "tfidf_best_cv.json")
    print("best config:", json.dumps(cfg), f"\nsaved to {args.out_config}")


if __name__ == "__main__":
    main()
