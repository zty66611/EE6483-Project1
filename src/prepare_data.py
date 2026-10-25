"""Clean the dataset and create a reproducible train/validation split."""

from __future__ import annotations

import argparse

from sklearn.model_selection import train_test_split

from src.data import CLEAN_COL, LABEL_COL, METRICS_DIR, PROCESSED_DIR, RAW_DIR, TEXT_COL, clean_text, load_raw, save_json


def main() -> None:
    """Prepare processed datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(RAW_DIR / "train.json"))
    parser.add_argument("--test", default=str(RAW_DIR / "test.json"))
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_raw(args.train)
    df[CLEAN_COL] = df[TEXT_COL].map(clean_text)
    n_raw = len(df)

    df = df[df[CLEAN_COL] != ""]
    n_empty = n_raw - len(df)

    # Deduplicate before splitting so identical texts cannot land in both train and validation.
    key = df[CLEAN_COL].str.lower()
    conflicting = df.groupby(key)[LABEL_COL].transform("nunique") > 1
    n_conflict = int(conflicting.sum())
    df = df[~conflicting]
    before = len(df)
    df = df[~df[CLEAN_COL].str.lower().duplicated(keep="first")]
    n_dup = before - len(df)

    train, val = train_test_split(
        df, test_size=args.val_size, stratify=df[LABEL_COL], random_state=args.seed
    )
    train, val = train.sort_values("orig_index"), val.sort_values("orig_index")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["orig_index", TEXT_COL, CLEAN_COL, LABEL_COL]
    train[cols].to_csv(PROCESSED_DIR / "train.csv", index=False)
    val[cols].to_csv(PROCESSED_DIR / "val.csv", index=False)
    df[cols].to_csv(PROCESSED_DIR / "full.csv", index=False)

    test = load_raw(args.test)
    test[CLEAN_COL] = test[TEXT_COL].map(clean_text)
    test[["orig_index", TEXT_COL, CLEAN_COL]].to_csv(PROCESSED_DIR / "test.csv", index=False)

    summary = {
        "seed": args.seed,
        "val_size": args.val_size,
        "raw_rows": n_raw,
        "dropped_empty": n_empty,
        "dropped_conflicting_label_rows": n_conflict,
        "dropped_duplicates": n_dup,
        "kept_rows": len(df),
        "train_rows": len(train),
        "val_rows": len(val),
        "train_label_counts": {int(k): int(v) for k, v in train[LABEL_COL].value_counts().sort_index().items()},
        "val_label_counts": {int(k): int(v) for k, v in val[LABEL_COL].value_counts().sort_index().items()},
        "test_rows": len(test),
    }
    # Indices into the raw train.json; tracked in git so every member can verify the same split.
    save_json({**summary, "train_index": train["orig_index"].tolist(), "val_index": val["orig_index"].tolist()},
              METRICS_DIR / "split_indices.json")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
