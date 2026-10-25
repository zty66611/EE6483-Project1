"""Train the TF-IDF + Logistic Regression baseline.

    python -m src.train_tfidf --config configs/e1_word.json --run-id E1_word
    python -m src.train_tfidf --config configs/tfidf_best.json --run-id final --full   # refit on train+val
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import joblib
import psutil
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data import CLEAN_COL, LABEL_COL, MODELS_DIR, load_json, load_split
from src.evaluate import compute_metrics, log_run, save_predictions


def build_pipeline(config: dict) -> Pipeline:
    """Vectorizer(s) and classifier in one Pipeline, so vocabulary and IDF are saved with the model."""
    clf_cfg = config["classifier"]
    if clf_cfg["type"] == "majority":
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])

    features = config.get("features", "word")
    word_cfg, char_cfg = config.get("vectorizer", {}), config.get("char_vectorizer", {})
    parts = []
    if "word" in features:
        parts.append(("word", TfidfVectorizer(
            lowercase=True,
            ngram_range=tuple(word_cfg.get("ngram_range", (1, 2))),
            min_df=word_cfg.get("min_df", 2),
            max_features=word_cfg.get("max_features"),
            sublinear_tf=word_cfg.get("sublinear_tf", True),
            token_pattern=r"(?u)\b\w+\b|[!?]",  # keep one-letter words and !/? ; no stop-word removal
        )))
    if "char" in features:
        parts.append(("char", TfidfVectorizer(
            lowercase=True,
            analyzer=char_cfg.get("analyzer", "char_wb"),
            ngram_range=tuple(char_cfg.get("ngram_range", (3, 5))),
            min_df=char_cfg.get("min_df", 2),
            max_features=char_cfg.get("max_features"),
            sublinear_tf=char_cfg.get("sublinear_tf", True),
        )))
    if not parts:
        raise ValueError(f"unknown features setting {features!r}")
    vec = parts[0][1] if len(parts) == 1 else FeatureUnion(parts)

    clf = LogisticRegression(
        C=clf_cfg.get("C", 1.0),
        max_iter=clf_cfg.get("max_iter", 1000),
        class_weight=clf_cfg.get("class_weight"),
        solver="liblinear",
        random_state=config.get("random_seed", 42),
    )
    return Pipeline([("tfidf", vec), ("clf", clf)])


class PeakRSS:
    """Sample this process's resident memory in a background thread and keep the maximum."""

    def __init__(self, interval: float = 0.02):
        self.proc, self.interval, self.peak = psutil.Process(os.getpid()), interval, 0
        self._stop = threading.Event()

    def __enter__(self):
        self.start = self.peak = self.proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


def n_features(pipe: Pipeline) -> int | None:
    if "tfidf" not in pipe.named_steps:
        return None
    vec = pipe.named_steps["tfidf"]
    if isinstance(vec, FeatureUnion):
        return sum(len(v.vocabulary_) for _, v in vec.transformer_list)
    return len(vec.vocabulary_)


def fit_and_evaluate(config: dict, train, val):
    """Fit on ``train``, score on ``val``; returns pipeline, metrics, resources, predictions, probabilities."""
    pipe = build_pipeline(config)
    with PeakRSS() as mem:
        t0 = time.perf_counter()
        pipe.fit(train[CLEAN_COL], train[LABEL_COL])
        train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    prob = pipe.predict_proba(val[CLEAN_COL])[:, 1]
    infer_time = time.perf_counter() - t0
    pred = (prob >= 0.5).astype(int) if config["classifier"]["type"] != "majority" else pipe.predict(val[CLEAN_COL])
    resources = {
        "train_time_s": train_time,
        "val_infer_time_s": infer_time,
        "peak_ram_mb": mem.peak / 2**20,
        "train_ram_increase_mb": (mem.peak - mem.start) / 2**20,
        "n_features": n_features(pipe),
    }
    return pipe, compute_metrics(val[LABEL_COL], pred), resources, pred, prob


def main() -> None:
    """Train and save the baseline model."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/tfidf.json")
    parser.add_argument("--run-id", help="defaults to the config file name")
    parser.add_argument("--full", action="store_true",
                        help="refit on the full deduplicated training data (train+val) for test prediction")
    args = parser.parse_args()

    config = load_json(args.config)
    run_id = args.run_id or Path(args.config).stem
    train, val = load_split("train"), load_split("val")

    pipe, metrics, resources, pred, prob = fit_and_evaluate(config, train, val)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{run_id}.joblib"
    joblib.dump(pipe, model_path)
    resources["model_size_mb"] = model_path.stat().st_size / 2**20
    resources["model_path"] = str(model_path.relative_to(MODELS_DIR.parents[1]))
    log_run(run_id, "tfidf_lr" if config["classifier"]["type"] != "majority" else "majority", config,
            metrics, resources)
    save_predictions(run_id, val, pred, prob)
    print(f"[{run_id}] val accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
          f"train={resources['train_time_s']:.2f}s size={resources['model_size_mb']:.2f}MB")

    if args.full:
        full = load_split("full")
        full_pipe = build_pipeline(config)
        t0 = time.perf_counter()
        full_pipe.fit(full[CLEAN_COL], full[LABEL_COL])
        full_path = MODELS_DIR / f"{run_id}_full.joblib"
        joblib.dump(full_pipe, full_path)
        print(f"[{run_id}] refit on {len(full)} rows in {time.perf_counter() - t0:.2f}s -> {full_path}")


if __name__ == "__main__":
    main()
