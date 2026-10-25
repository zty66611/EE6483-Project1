"""Generate test-set predictions and outputs/submission.csv.

    python -m src.predict --model outputs/models/tfidf_best_full.joblib --output outputs/submission.csv
    python -m src.predict --model ... --template data/raw/sample_submission.csv   # once a template is provided

Without a template the file has columns ``reviews,sentiments`` in the original test.json order
(the course has not supplied a sample submission yet; confirm the expected format).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import CLEAN_COL, LABEL_COL, OUTPUTS_DIR, RAW_DIR, TEXT_COL, clean_text, load_raw


def predict_labels(model_path: Path, texts: pd.Series) -> np.ndarray:
    if model_path.suffix == ".joblib":
        return joblib.load(model_path).predict(texts).astype(int)
    # A directory is a Hugging Face checkpoint; implemented alongside src/train_transformer.py.
    raise NotImplementedError(f"No predictor for {model_path}; add the transformer branch here.")


def check_submission(sub: pd.DataFrame, n_test: int, template: pd.DataFrame | None, pred_col: str) -> None:
    """Assert row count, column names, ID order, label range and absence of NaN / index columns."""
    assert len(sub) == n_test, f"row count {len(sub)} != test size {n_test}"
    assert sub[pred_col].notna().all(), "NaN predictions"
    assert set(sub[pred_col].unique()) <= {0, 1}, "labels outside {0, 1}"
    assert sub[pred_col].dtype.kind == "i", "labels must be integers"
    assert not any(str(c).startswith("Unnamed") for c in sub.columns), "stray index column"
    if template is not None:
        assert list(sub.columns) == list(template.columns), "columns differ from template"
        other = [c for c in template.columns if c != pred_col]
        assert sub[other].equals(template[other]), "ID / non-prediction columns changed"


def main() -> None:
    """Load a trained model and generate predictions."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", default=str(RAW_DIR / "test.json"))
    parser.add_argument("--output", default=str(OUTPUTS_DIR / "submission.csv"))
    parser.add_argument("--template", help="course sample submission; only its prediction column is replaced")
    parser.add_argument("--pred-col", default=LABEL_COL)
    args = parser.parse_args()

    test = load_raw(args.test)
    preds = predict_labels(Path(args.model), test[TEXT_COL].map(clean_text))

    if args.template:
        template = pd.read_csv(args.template)
        sub = template.copy()
        sub[args.pred_col] = preds
    else:
        template = None
        sub = pd.DataFrame({TEXT_COL: test[TEXT_COL], args.pred_col: preds})

    check_submission(sub, len(test), template, args.pred_col)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)
    # Round-trip check: what is written must read back identically.
    check_submission(pd.read_csv(args.output, keep_default_na=False), len(test), template, args.pred_col)
    counts = sub[args.pred_col].value_counts().sort_index().to_dict()
    print(f"wrote {args.output}: {len(sub)} rows, label counts {counts}; all checks passed")


if __name__ == "__main__":
    main()
