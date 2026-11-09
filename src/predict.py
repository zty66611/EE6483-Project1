"""使用训练好的 DistilBERT 对产品评论测试集预测，并生成课程提交文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 项目根目录由当前脚本的位置推导，避免用户从其他工作目录运行时找不到文件。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "distilbert-seed62"
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "metrics" / "distilbert_seed62.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "distilbert.json"
DEFAULT_TEST = PROJECT_ROOT / "data" / "raw" / "test.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "submission.csv"


def load_json(path: Path) -> dict:
    """读取 UTF-8 JSON 配置或实验指标文件。"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_threshold(metrics_path: Path, override: float | None) -> float:
    """优先使用命令行阈值；否则读取所选 checkpoint 对应的验证集最佳阈值。"""
    if override is not None:
        threshold = override
    else:
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"未找到阈值记录 {metrics_path}；请提供 --metrics 或显式指定 --threshold。"
            )
        metrics = load_json(metrics_path)
        # 兼容训练脚本保存的顶层 best_threshold，以及嵌套指标中的 threshold。
        threshold = metrics.get("best_threshold")
        if threshold is None:
            threshold = metrics.get("val_metrics", {}).get("threshold")
        if threshold is None:
            raise ValueError(f"{metrics_path} 中没有找到验证集分类阈值。")

    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"分类阈值必须在 0 到 1 之间，当前为 {threshold}。")
    return threshold


@torch.inference_mode()
def predict_probabilities(
    texts: list[str],
    model_dir: Path,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """逐批预测正面概率；shuffle=False 保证输出顺序与输入评论一致。"""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_dir}")

    # tokenizer 必须和该 checkpoint 一起保存，确保分词词表完全匹配。
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )
    if model.config.num_labels != 2:
        raise ValueError(f"模型应为二分类模型，实际 num_labels={model.config.num_labels}。")

    # 训练时使用 max_length 截断，并将样本补齐到固定长度；预测采用相同处理方式。
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
    batches = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 优先使用 CUDA；如果没有可用显卡则自动退回 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    probabilities: list[np.ndarray] = []

    for input_ids, attention_mask in batches:
        # 将当前批次送到模型所在设备，并取类别 1（正面）的 softmax 概率。
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).logits
        positive_probs = torch.softmax(logits, dim=-1)[:, 1]
        probabilities.append(positive_probs.cpu().numpy())

    if not probabilities:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(probabilities)


def validate_submission(
    submission: pd.DataFrame,
    expected_rows: int,
    prediction_column: str,
    template: pd.DataFrame | None,
) -> None:
    """检查提交文件的行数、列、标签范围、缺失值和模板字段是否正确。"""
    if len(submission) != expected_rows:
        raise ValueError(f"提交行数 {len(submission)} 与测试集 {expected_rows} 不一致。")
    if prediction_column not in submission.columns:
        raise ValueError(f"提交文件缺少预测列 {prediction_column!r}。")
    if submission[prediction_column].isna().any():
        raise ValueError("预测列包含空值。")
    if not set(submission[prediction_column].unique()).issubset({0, 1}):
        raise ValueError("预测标签只能是 0 或 1。")
    if not pd.api.types.is_integer_dtype(submission[prediction_column]):
        raise ValueError("预测标签必须是整数类型。")
    if any(str(column).startswith("Unnamed") for column in submission.columns):
        raise ValueError("提交文件包含意外的索引列。")

    # 有课程模板时，要求列顺序完全相同，并检查所有非预测列都未改变。
    if template is not None:
        if list(submission.columns) != list(template.columns):
            raise ValueError("提交文件列名或列顺序与课程模板不一致。")
        unchanged_columns = [column for column in template.columns if column != prediction_column]
        if not submission[unchanged_columns].equals(template[unchanged_columns]):
            raise ValueError("提交文件中的 ID 或其他模板字段发生改变。")


def main() -> None:
    """加载 seed 62 最佳模型，读取测试评论并写出 submission.csv。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR, help="Hugging Face checkpoint 目录")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="checkpoint 对应的验证指标 JSON")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="训练配置，用于读取 max_length")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST, help="原始测试集 JSON 文件")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 CSV 路径")
    parser.add_argument("--template", type=Path, help="可选的课程提交模板 CSV")
    parser.add_argument("--pred-col", default="label", help="提交文件中的预测列名；课程示例为 label")
    parser.add_argument(
        "--id-from-row",
        action="store_true",
        help="测试集没有 ID 且没有模板时，以 1 到样本数的行号生成 id；仅在确认课程接受行号 ID 后使用",
    )
    parser.add_argument("--threshold", type=float, help="手动覆盖阈值；默认读取验证集选定阈值")
    parser.add_argument("--batch-size", type=int, default=32, help="预测批次大小")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size 必须是正整数。")
    for path, label in ((args.config, "训练配置"), (args.test, "测试集")):
        if not path.is_file():
            raise FileNotFoundError(f"{label}文件不存在：{path}")

    config = load_json(args.config)
    max_length = int(config.get("max_length", 256))
    if max_length <= 0:
        raise ValueError("max_length 必须是正整数。")
    threshold = resolve_threshold(args.metrics, args.threshold)

    # test.json 是 JSON 数组；转换为 DataFrame 后仍保持文件中的原始样本顺序。
    test_df = pd.read_json(args.test)
    if "reviews" not in test_df.columns:
        raise ValueError("测试集缺少必需的 reviews 文本列。")
    # 空评论作为空字符串处理，避免把缺失值错误转成字符串 'nan'。
    texts = test_df["reviews"].fillna("").astype(str).tolist()

    print(f"[*] 模型：{args.model}")
    print(f"[*] 测试样本：{len(texts)}，最大长度：{max_length}，阈值：{threshold:.2f}")
    probabilities = predict_probabilities(texts, args.model, max_length, args.batch_size)
    predictions = (probabilities >= threshold).astype(np.int64)

    # 优先使用课程模板中已有的 ID；模板只负责确定提交格式和 ID 顺序。
    template = None
    if args.template is not None:
        if not args.template.is_file():
            raise FileNotFoundError(f"提交模板不存在：{args.template}")
        template = pd.read_csv(args.template)
        if len(template) != len(test_df):
            raise ValueError("课程模板行数与测试集行数不一致。")
        if "id" not in template.columns:
            raise ValueError("课程模板缺少 id 列；请使用包含 id 和 label 的模板。")
        submission = template.copy()
        submission[args.pred_col] = predictions
    else:
        # 有些测试文件直接包含 ID；若没有，必须显式同意按原始行号生成 ID。
        if "id" in test_df.columns:
            ids = test_df["id"]
        elif args.id_from_row:
            ids = np.arange(1, len(test_df) + 1, dtype=np.int64)
        else:
            raise ValueError(
                "PDF 示例要求 id,label，但测试 JSON 没有 id。请提供 --template "
                "或确认课程允许行号 ID 后使用 --id-from-row。"
            )
        submission = pd.DataFrame({"id": ids, args.pred_col: predictions})

    # 按 PDF 示例保持两列结构和固定列名，防止把评论文本误当成提交 ID。
    if list(submission.columns) != ["id", args.pred_col]:
        raise ValueError(
            f"提交列应为 ['id', '{args.pred_col}']，实际为 {list(submission.columns)}。"
        )

    validate_submission(submission, len(test_df), args.pred_col, template)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, encoding="utf-8-sig")

    # 重新读取最终 CSV 再检查一次，确认磁盘上的交付文件也符合格式要求。
    written = pd.read_csv(args.output, keep_default_na=False)
    validate_submission(written, len(test_df), args.pred_col, template)
    counts = written[args.pred_col].value_counts().sort_index().to_dict()
    print(f"[+] 已生成 {args.output}：{len(written)} 行，标签分布 {counts}；检查通过。")


if __name__ == "__main__":
    main()
