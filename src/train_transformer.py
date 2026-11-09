# ============================================================================

# ============================================================================


import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from src.evaluate import compute_metrics, log_run, save_predictions


# ============================================================================

# ============================================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


LOCAL_PRETRAINED = MODEL_DIR / "_pretrained" / "distilbert-base-uncased"

# ============================================================================

# ============================================================================

def load_config(config_path: Path) -> dict:


    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"[*] 已读取配置: {config_path}")
    for key in ["model_name", "max_length", "learning_rate", "batch_size", "epochs"]:
        if key in cfg:
            print(f"    {key:16s} = {cfg[key]}")
    return cfg


def resolve_model_path(cfg: dict) -> str:


    if LOCAL_PRETRAINED.is_dir():
        print(f"[*] 使用本地预训练权重: {LOCAL_PRETRAINED}")
        return str(LOCAL_PRETRAINED)

    print("[!] 本地未找到预训练权重，将按模型名联网加载:", cfg["model_name"])
    return cfg["model_name"]


# ============================================================================

# ============================================================================

def load_data():


    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")


    for df in (train_df, val_df):
        df["reviews"] = df["reviews"].astype(str)
    train_df = train_df.dropna(subset=["reviews", "sentiments"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=["reviews", "sentiments"]).reset_index(drop=True)


    print(f"[*] 训练集: {len(train_df)} 条  "
          f"(好评 1: {(train_df['sentiments'] == 1).sum()}, "
          f"差评 0: {(train_df['sentiments'] == 0).sum()})")
    print(f"[*] 验证集: {len(val_df)} 条  "
          f"(好评 1: {(val_df['sentiments'] == 1).sum()}, "
          f"差评 0: {(val_df['sentiments'] == 0).sum()})")

    return train_df, val_df


# ============================================================================

# ============================================================================

def build_dataset(tokenizer, df, max_length: int) -> TensorDataset:


    encoded = tokenizer(
        list(df["reviews"]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = torch.tensor(df["sentiments"].values, dtype=torch.long)

    return TensorDataset(input_ids, attention_mask, labels)


# ============================================================================

# ============================================================================

def compute_class_weights(train_df) -> torch.Tensor:


    counts = train_df["sentiments"].value_counts().sort_index()
    n_total = len(train_df)
    n_classes = 2


    weights = n_total / (n_classes * counts.values.astype("float32"))

    print(f"[*] 类别权重: 差评(0) = {weights[0]:.3f}, 好评(1) = {weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================

# ============================================================================

def metrics_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:


    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")


    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )


    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "negative_precision": float(precision[0]),
        "negative_recall": float(recall[0]),
        "negative_f1": float(f1[0]),
        "positive_precision": float(precision[1]),
        "positive_recall": float(recall[1]),
        "positive_f1": float(f1[1]),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def predict_probs(model, data_loader, device) -> tuple:


    model.eval()

    all_probs = []
    all_labels = []
    all_logits = []

    for input_ids, attention_mask, labels in data_loader:

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)


        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits


        probs = torch.softmax(logits, dim=-1)[:, 1]

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_logits.append(logits.cpu().numpy())


    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_logits),
    )


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple:


    best_threshold, best_f1 = 0.5, -1.0

    for t in np.arange(0.05, 0.96, 0.01):
        score = f1_score(labels, (probs >= t).astype(int), average="macro")
        if score > best_f1:
            best_f1, best_threshold = score, float(t)

    return best_threshold, best_f1


# ============================================================================

# ============================================================================

def plot_curves(history: dict, save_path: Path) -> None:


    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))


    axes[0].plot(epochs, history["train_loss"], marker="o", label="train loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)


    axes[1].plot(epochs, history["val_macro_f1"], marker="o", color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Validation Macro-F1")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[+] 训练曲线已保存: {save_path}")


# ============================================================================

# ============================================================================

def train(cfg: dict) -> None:


    seed = int(cfg.get("random_seed", 42))
    run_name = str(cfg.get("run_name", f"seed{seed}")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name may contain only letters, numbers, hyphens, and underscores")
    best_model_dir = MODEL_DIR / f"distilbert-{run_name}"
    metrics_path = METRICS_DIR / f"distilbert_{run_name}.json"
    curve_path = FIGURES_DIR / f"distilbert_training_curves_{run_name}.png"
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[*] 使用 GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
    else:
        print("[!] 没检测到 GPU，会用 CPU 训练（会非常慢，仅供调试）")


    max_length = cfg.get("max_length", 256)
    batch_size = cfg.get("batch_size", 16)
    epochs = cfg.get("epochs", 3)
    learning_rate = float(cfg.get("learning_rate", 2e-5))
    weight_decay = float(cfg.get("weight_decay", 0.01))

    train_df, val_df = load_data()


    model_path = resolve_model_path(cfg)
    print("[*] 正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"[*] 正在分词（max_length={max_length}）...")
    train_dataset = build_dataset(tokenizer, train_df, max_length)
    val_dataset = build_dataset(tokenizer, val_df, max_length)


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )


    print("[*] 正在加载模型...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] 模型参数量: {n_params / 1e6:.1f}M")


    use_class_weight = cfg.get("use_class_weight", True)
    class_weights = compute_class_weights(train_df).to(device) if use_class_weight else None
    if class_weights is None:
        print("[*] 未启用类别加权（use_class_weight=false）")


    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5

    train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()


        model.train()
        running_loss = 0.0

        for step, (input_ids, attention_mask, labels) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()


            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits


            loss = F.cross_entropy(logits, labels, weight=class_weights)


            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


            optimizer.step()

            running_loss += loss.item()


        avg_train_loss = running_loss / len(train_loader)


        val_probs, val_labels, val_logits = predict_probs(model, val_loader, device)


        val_loss = F.cross_entropy(
            torch.from_numpy(val_logits), torch.from_numpy(val_labels)
        ).item()


        val_metrics_05 = metrics_from_probs(val_probs, val_labels, threshold=0.5)


        tuned_threshold, tuned_f1 = find_best_threshold(val_probs, val_labels)


        if tuned_f1 > val_metrics_05["macro_f1"]:
            val_metrics = metrics_from_probs(val_probs, val_labels, threshold=tuned_threshold)
        else:
            val_metrics = val_metrics_05

        val_macro_f1 = val_metrics["macro_f1"]


        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_macro_f1)

        epoch_time = time.time() - epoch_start
        print(f"  epoch {epoch}/{epochs} | train_loss {avg_train_loss:.4f} "
              f"| val_loss {val_loss:.4f} "
              f"| val_acc {val_metrics['accuracy']:.4f} "
              f"| val_macroF1 {val_macro_f1:.4f} "
              f"| threshold {val_metrics['threshold']:.2f} "
              f"| {epoch_time:.0f}s")


        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_threshold = val_metrics["threshold"]

            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"    -> 新的最佳模型已保存 (macro-F1 {best_macro_f1:.4f})")


    total_time = time.time() - train_start

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        peak_vram_mb = 0.0


    model_size_mb = sum(
        f.stat().st_size for f in best_model_dir.glob("*") if f.is_file()
    ) / 1024 ** 2

    print("\n" + "=" * 60)
    print(f"训练完成 | 最佳 epoch {best_epoch} | 最佳 Macro-F1 {best_macro_f1:.4f}")
    print(f"总耗时 {total_time / 60:.1f} 分钟 | 峰值显存 {peak_vram_mb:.0f} MB "
          f"| 模型大小 {model_size_mb:.0f} MB")
    print("=" * 60)


    METRICS_DIR.mkdir(parents=True, exist_ok=True)


    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(device)
    inference_start = time.perf_counter()
    final_probs, final_labels, _ = predict_probs(best_model, val_loader, device)
    val_infer_time = time.perf_counter() - inference_start
    final_metrics = metrics_from_probs(final_probs, final_labels, threshold=best_threshold)
    threshold_05_metrics = metrics_from_probs(final_probs, final_labels, threshold=0.5)
    common_metrics = compute_metrics(final_labels, (final_probs >= 0.5).astype(int))

    split_path = METRICS_DIR / "split_indices.json"
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest() if split_path.is_file() else None

    metrics_to_save = {
        "model": "distilbert-base-uncased",
        "config": cfg,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "total_seconds": round(total_time, 1),
        "peak_vram_mb": round(float(peak_vram_mb), 1),
        "model_size_mb": round(float(model_size_mb), 1),
        "n_params": int(n_params),
        "history": history,
        "val_metrics": final_metrics,
        "val_metrics_threshold_05": threshold_05_metrics,
        "split_sha256": split_sha256,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)
    print(f"[+] 指标已保存: {metrics_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_curves(history, curve_path)
    run_id = f"distilbert_{run_name}"
    resources = {
        "train_time_s": round(total_time, 3),
        "val_infer_time_s": round(val_infer_time, 3),
        "peak_gpu_mem_mb": round(float(peak_vram_mb), 3),
        "model_size_mb": round(float(model_size_mb), 3),
    }
    log_run(run_id, "distilbert", cfg, common_metrics, resources)
    save_predictions(run_id, val_df, (final_probs >= 0.5).astype(int), final_probs)
    print(f"Best checkpoint: {best_model_dir}")
    print(f"Training curve: {curve_path}")


# ============================================================================

# ============================================================================

def main() -> None:


    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for sentiment classification")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "distilbert.json"),
        help="配置文件路径，默认 configs/distilbert.json",
    )
    parser.add_argument("--seed", type=int, help="Override the random seed in the config")
    parser.add_argument("--run-name", type=str, help="Unique name used for this run's outputs")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.seed is not None:
        cfg["random_seed"] = args.seed
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    train(cfg)


if __name__ == "__main__":


    main()
