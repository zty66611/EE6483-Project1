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
python -m src.inspect_data --train data/raw/train.json --test data/raw/test.json
python -m src.train_tfidf --config configs/tfidf.json
python -m src.train_transformer --config configs/distilbert.json
python -m src.predict --model outputs/models/best --output outputs/submission.csv
```

> 当前 `src/` 中为任务入口骨架；三位成员应共用同一数据划分、随机种子、指标函数和输出目录。
