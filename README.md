# EE6483 Project 1

Product Review Sentiment Classification（产品评论情感二分类）。

## Project structure

```text
data/
  raw/                  # 原始训练集、测试集、提交模板（不提交数据文件）
  processed/            # 固定的训练/验证划分（不提交生成文件）
src/
  inspect_data.py       # 数据审计
  prepare_data.py       # 清洗与数据划分
  train_tfidf.py        # TF-IDF + Logistic Regression 基线
  train_transformer.py  # DistilBERT 微调
  evaluate.py           # 指标计算与图表生成
  predict.py            # 测试集预测
configs/
  tfidf.json            # 逻辑回归实验配置
  distilbert.json        # DistilBERT 实验配置
outputs/
  models/               # 模型文件
  metrics/              # 指标与实验表
  figures/              # 报告图表
  submission.csv        # 最终提交文件（运行预测后生成）
report/                 # 报告正文与参考资料
requirements.txt
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Main commands

```bash
# 1. 数据审计 -> outputs/metrics/data_audit.json, outputs/figures/{class,length}_distribution.png
python -m src.inspect_data
# 2. 清洗、去重、分层 80/20 划分 (seed 42) -> data/processed/{train,val,full,test}.csv
#    划分索引同时写入 outputs/metrics/split_indices.json，三人可核对是否一致
python -m src.prepare_data
# 3. 基线实验 E0 / E1 / E2
python -m src.train_tfidf --config configs/e0_majority.json  --run-id E0_majority
python -m src.train_tfidf --config configs/e1_word.json      --run-id E1_word
python -m src.train_tfidf --config configs/e2_word_char.json --run-id E2_word_char
# 4. 控制变量调参 + 5 折交叉验证 -> configs/tfidf_best.json, outputs/metrics/tfidf_tuning.*
python -m src.tune_tfidf
# 5. 最佳配置：验证集评估，并在 train+val 上重训
python -m src.train_tfidf --config configs/tfidf_best.json --run-id E2_best --full
# 6. 测试集预测（含格式自动检查）
python -m src.predict --model outputs/models/E2_best_full.joblib --output outputs/submission.csv
# 7. 汇总比较表与图 -> outputs/metrics/comparison.md, outputs/figures/model_comparison.png
python -m src.evaluate

python -m src.train_transformer --config configs/distilbert.json
```

## Shared conventions

- 所有模型读取 `src.data.load_split("train" | "val" | "full")`，使用 `text` 列（已清洗）和 `sentiments` 标签。
- 指标统一调用 `src.evaluate.compute_metrics`；每次运行用 `src.evaluate.log_run` 记录，
  会写入 `outputs/metrics/runs/<run_id>.json`、追加到 `outputs/metrics/experiments.csv`，并生成 `outputs/figures/cm_<run_id>.png`。
- 验证集预测用 `src.evaluate.save_predictions` 保存到 `outputs/metrics/predictions/<run_id>_val.csv`
  （列为 orig_index, text, sentiments, pred, prob_positive），供错误案例分析对比使用。
- 课程尚未提供 sample submission；拿到后用 `--template` 参数生成提交文件。
