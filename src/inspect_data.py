"""Audit raw review data: schema, labels, missing values, duplicates, and length distribution."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import CLEAN_COL, FIGURES_DIR, LABEL_COL, METRICS_DIR, RAW_DIR, TEXT_COL, clean_text, load_raw, save_json


def length_stats(words: pd.Series) -> dict:
    return {
        "mean": round(float(words.mean()), 2),
        "median": float(words.median()),
        "p95": float(words.quantile(0.95)),
        "max": int(words.max()),
    }


def audit(df: pd.DataFrame, name: str) -> dict:
    text = df[TEXT_COL]
    cleaned = text.map(clean_text)
    words = cleaned.str.split().str.len()
    report = {
        "split": name,
        "n_rows": len(df),
        "columns": [c for c in df.columns if c != "orig_index"],
        "missing_text": int(text.isna().sum()),
        "empty_text": int((cleaned == "").sum()),
        "exact_duplicate_rows": int(text.duplicated().sum()),
        "duplicate_after_cleaning": int(cleaned.str.lower().duplicated().sum()),
        "html_remnants": int(text.str.contains(r"<[^>]+>|&[a-z]+;", regex=True).sum()),
        "words": length_stats(words),
        "chars": length_stats(cleaned.str.len()),
    }
    if LABEL_COL in df:
        counts = df[LABEL_COL].value_counts().sort_index()
        report["label_counts"] = {int(k): int(v) for k, v in counts.items()}
        report["label_ratio"] = {int(k): round(v / len(df), 4) for k, v in counts.items()}
        report["missing_label"] = int(df[LABEL_COL].isna().sum())
        groups = df.assign(_k=cleaned.str.lower()).groupby("_k")[LABEL_COL].nunique()
        report["texts_with_conflicting_labels"] = int((groups > 1).sum())
        report["words_by_label"] = {
            int(k): length_stats(words[df[LABEL_COL] == k]) for k in counts.index
        }
    return report


def plot_class_distribution(train: pd.DataFrame) -> None:
    counts = train[LABEL_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(["Negative (0)", "Positive (1)"], counts.values, color=["#c0504d", "#4f81bd"])
    ax.bar_label(bars, labels=[f"{v} ({v / counts.sum():.1%})" for v in counts.values])
    ax.set_ylabel("Number of reviews")
    ax.set_title("Training set class distribution")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)


def plot_length_distribution(train: pd.DataFrame, test: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = range(0, 401, 10)
    for label, name, color in [(0, "Train negative", "#c0504d"), (1, "Train positive", "#4f81bd")]:
        words = train.loc[train[LABEL_COL] == label, CLEAN_COL].str.split().str.len()
        ax.hist(words.clip(upper=400), bins=bins, alpha=0.55, density=True, label=name, color=color)
    test_words = test[CLEAN_COL].str.split().str.len()
    ax.hist(test_words.clip(upper=400), bins=bins, density=True, histtype="step", lw=1.5,
            label="Test", color="black")
    ax.set_xlabel("Words per review (clipped at 400)")
    ax.set_ylabel("Density")
    ax.set_title("Review length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "length_distribution.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Run data-quality inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(RAW_DIR / "train.json"))
    parser.add_argument("--test", default=str(RAW_DIR / "test.json"))
    parser.add_argument("--samples", type=int, default=10, help="reviews per class to print for manual check")
    args = parser.parse_args()

    train, test = load_raw(args.train), load_raw(args.test)
    report = {"train": audit(train, "train"), "test": audit(test, "test")}
    train_keys = set(train[TEXT_COL].map(clean_text).str.lower())
    report["test_texts_also_in_train"] = int(test[TEXT_COL].map(clean_text).str.lower().isin(train_keys).sum())

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_json(report, METRICS_DIR / "data_audit.json")

    for df in (train, test):
        df[CLEAN_COL] = df[TEXT_COL].map(clean_text)
    plot_class_distribution(train)
    plot_length_distribution(train, test)

    import json
    print(json.dumps(report, indent=2))
    for label in sorted(train[LABEL_COL].unique()):
        print(f"\n--- {args.samples} random reviews with label {label} ---")
        for t in train.loc[train[LABEL_COL] == label, CLEAN_COL].sample(args.samples, random_state=42):
            print("*", t[:200])
    print(f"\nSaved {METRICS_DIR / 'data_audit.json'} and figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
