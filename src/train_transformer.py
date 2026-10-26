"""
DistilBERT 情感二分类微调脚本（产品评论 → 好评 / 差评）
=========================================================

这个文件做的事情，用一句话说：
    把已经训练好的 DistilBERT 模型"接上一个小分类头"，再用我们自己的
    5748 条带标签评论把它微调（fine-tune）成一个情感分类器。

打个比方：
    DistilBERT 是一个读过大量英文（Wikipedia、书籍）的"语感很好的人"，
    但它不知道该把评论分成好评还是差评。我们做的事情就是给它一个小测验
    （带标签的数据），让它把已有的语言能力"对准"我们的任务。这个过程
    只需要很少的数据和很短的训练，这叫微调（fine-tuning）。

怎么运行（在项目根目录下）：
    E:\\Anaconda\\envs\\pytorch-gpu\\python.exe -m src.train_transformer --config configs/distilbert.json

跑完之后会生成三样东西：
    outputs/models/distilbert-best/        验证集上表现最好的模型（可以直接拿去预测测试集）
    outputs/metrics/distilbert_metrics.json 所有指标数字（写报告时用这些数）
    outputs/figures/distilbert_training_curves.png 训练曲线图（报告里的图）

作者：梁晨东（EE6483 小组 B，负责 DistilBERT 部分）
"""

# ============================================================================
# 第 0 步：导入需要用到的库
# ============================================================================
# 每个 import 都是**标准写法**，不需要背，知道大概负责什么就行。

import argparse      # 解析命令行参数（让我们能用 --config 指定配置文件）
import json          # 读写 json 文件（配置文件、指标文件都是 json 格式）
import time          # 计时（记录训练花了多久，报告里要写资源消耗）
from pathlib import Path  # 处理文件路径，比字符串拼接更安全

import numpy as np   # 数值计算
import pandas as pd  # 读取 csv 数据表
import torch         # PyTorch 深度学习框架（跑模型的那一层）

import torch.nn.functional as F  # 里面有很多现成的"函数式"工具，比如交叉熵损失
from sklearn.metrics import (    # scikit-learn 的评估指标
    accuracy_score,              # 准确率
    confusion_matrix,            # 混淆矩阵
    f1_score,                    # F1 分数
    precision_recall_fscore_support,  # 一次算出每一类的精确率/召回率/F1
)
from torch.utils.data import DataLoader, TensorDataset  # 数据分批与打包
from transformers import (       # HuggingFace 的库：模型和分词器都从这里来
    AutoModelForSequenceClassification,  # 带分类头的预训练模型
    AutoTokenizer,                       # 分词器（把文字变成数字 ID）
)


# ============================================================================
# 第 1 步：定义各种路径
# ============================================================================
# Path(__file__) 是"当前这个 py 文件的路径"，.parent 是它所在的文件夹（src/），
# 再 .parent 一次就是项目根目录。这样写的好处是：不管你在哪个目录下运行脚本，
# 路径都能找对，不会出现"我在 A 文件夹运行就找不到数据"的问题。

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"        # 已经切分好的 train/val/test
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"       # 模型保存位置
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"    # 指标保存位置
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"    # 图片保存位置

# 预训练权重的本地目录。
# 注意：本机 huggingface_hub 建符号链接会失败，用模型名在线加载会报 tokenizer 错误，
# 所以权重已经提前下载到项目里，代码优先读这个本地路径。
LOCAL_PRETRAINED = MODEL_DIR / "_pretrained" / "distilbert-base-uncased"

BEST_MODEL_DIR = MODEL_DIR / "distilbert-best"        # 最终保存的最佳模型


# ============================================================================
# 第 2 步：读配置
# ============================================================================

def load_config(config_path: Path) -> dict:
    """
    读取 json 配置文件。

    configs/distilbert.json 里存的是超参数（学习率、batch size 等）。
    把这些东西单独放在配置文件里，好处是：想换参数不用改代码，
    而且做实验时"哪次用了什么参数"有记录，报告里可以追溯。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"[*] 已读取配置: {config_path}")
    for key in ["model_name", "max_length", "learning_rate", "batch_size", "epochs"]:
        if key in cfg:
            print(f"    {key:16s} = {cfg[key]}")
    return cfg


def resolve_model_path(cfg: dict) -> str:
    """
    决定"从哪儿加载预训练模型"。

    - 如果本地有下载好的权重目录 → 用本地（快、稳定，本机唯一可行的方式）
    - 否则 → 退回用配置里的模型名，让 transformers 自己联网下载
    """
    if LOCAL_PRETRAINED.is_dir():
        print(f"[*] 使用本地预训练权重: {LOCAL_PRETRAINED}")
        return str(LOCAL_PRETRAINED)

    print("[!] 本地未找到预训练权重，将按模型名联网加载:", cfg["model_name"])
    return cfg["model_name"]


# ============================================================================
# 第 3 步：读数据
# ============================================================================

def load_data():
    """
    读入训练集和验证集。

    这两份 csv 是李曜（成员 A）用 prepare_data.py 切好的，
    已经做过去重和分层抽样（正负比例一致），我们直接拿来用，
    **不要自己再切一次**，否则三个人的结果没法比较。
    """
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    # 保险起见，把评论统一转成字符串，并去掉可能存在的空值
    for df in (train_df, val_df):
        df["reviews"] = df["reviews"].astype(str)
    train_df = train_df.dropna(subset=["reviews", "sentiments"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=["reviews", "sentiments"]).reset_index(drop=True)

    # 打印一下正负样本数量，方便随时确认数据没读错
    print(f"[*] 训练集: {len(train_df)} 条  "
          f"(好评 1: {(train_df['sentiments'] == 1).sum()}, "
          f"差评 0: {(train_df['sentiments'] == 0).sum()})")
    print(f"[*] 验证集: {len(val_df)} 条  "
          f"(好评 1: {(val_df['sentiments'] == 1).sum()}, "
          f"差评 0: {(val_df['sentiments'] == 0).sum()})")

    return train_df, val_df


# ============================================================================
# 第 4 步：分词 + 打包成 PyTorch 能吃的格式
# ============================================================================

def build_dataset(tokenizer, df, max_length: int) -> TensorDataset:
    """
    把"一串英文评论"变成"一堆数字矩阵"。

    这是整个流程里最关键的一步，展开讲：
    1) 分词（tokenize）：模型不认识单词，只认识数字。分词器把
       "good product" 拆成 ["good", "product"] 再查表变成 [2204, 3105]。
    2) 截断（truncation）：模型最多只能看 max_length 个词，太长的部分直接切掉。
    3) 补齐（padding）：一个 batch 里每条评论长度不一样没法堆成矩阵，
       所以要统一补到 max_length（补的占位符叫 [PAD]）。
    4) attention_mask：告诉模型"哪些位置是真的词（1），哪些是我补的 [PAD]（0）"，
       不然模型会把填充的空位也当成内容去读，结果就错了。

    返回的 TensorDataset 就像三个一一对应的列表：
        input_ids     每个词的编号       形状 [样本数, max_length]
        attention_mask 哪些位置是真词     形状 [样本数, max_length]
        labels        0 或 1             形状 [样本数]
    """
    # return_tensors="pt" 表示直接返回 PyTorch 张量（而不是 python 列表）
    encoded = tokenizer(
        list(df["reviews"]),
        padding="max_length",   # 全部补/截到 max_length
        truncation=True,        # 超长就截断
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]          # 词的数字编号
    attention_mask = encoded["attention_mask"]  # 0/1 掩码
    labels = torch.tensor(df["sentiments"].values, dtype=torch.long)  # 标签必须是整数

    return TensorDataset(input_ids, attention_mask, labels)


# ============================================================================
# 第 5 步：处理"类别不平衡"——本项目最重要的一个细节
# ============================================================================

def compute_class_weights(train_df) -> torch.Tensor:
    """
    算出每个类别的损失权重。

    为什么要这个？
        我们的数据里好评 4909 条、差评只有 839 条（大约 85% : 15%）。
        如果不管它，模型只要"无脑全猜好评"就能拿到 85% 的准确率——
        数字好看，但完全没用（差评一个都抓不到）。
        这叫类别不平衡（class imbalance）。

    怎么解决？
        给"样本少的类别"更高的权重，让模型犯一次差评的错，代价更大。
        公式（逆频率加权）：weight_i = 总样本数 / (类别数 × 该类样本数)
        这样正负两类在损失里的"总贡献"就被拉平了。

    注意：权重只在训练时用于计算损失，评估时不用（评估要反映真实情况）。
    """
    # value_counts() 统计每类多少条；sort_index() 保证顺序是 [0 类, 1 类]
    counts = train_df["sentiments"].value_counts().sort_index()
    n_total = len(train_df)
    n_classes = 2

    # 逆频率：少数类权重大，多数类权重小
    weights = n_total / (n_classes * counts.values.astype("float32"))

    print(f"[*] 类别权重: 差评(0) = {weights[0]:.3f}, 好评(1) = {weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# 第 6 步：评估函数
# ============================================================================

def metrics_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """
    从"模型给出的概率"算出各种指标。

    参数说明：
        probs   模型认为"这条是好评"的概率，0~1 之间，形状 [样本数]
        labels  真实标签（0 或 1）
        threshold 判定阈值：概率 >= threshold 就判成好评。
                  默认 0.5，但因为数据不平衡，0.5 往往不是最优的，
                  所以下面还会额外搜一个更好的阈值。

    为什么主指标用 Macro-F1 而不是 accuracy？
        Macro-F1 是"分别算好/差两类的 F1，再取平均"，
        两类地位平等。模型如果只顾多数类，Macro-F1 会立刻掉下来，
        所以它能真实反映模型有没有学会区分差评。
    """
    # 概率 >= 阈值 → 预测为 1（好评），否则 0（差评）
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    # 每一类的精确率(precision)、召回率(recall)、F1
    # labels=[0,1] 保证即使某类一个都没预测出来，也按 0/1 两类的顺序返回
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )

    # 混淆矩阵：行是真实类别，列是预测类别
    # [[真差评判成差评, 真差评判成好评],
    #  [真好评选成差评, 真好评选成好评]]
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


@torch.no_grad()   # 这个装饰器表示"这段代码不需要算梯度"，能省显存、跑得更快
def predict_probs(model, data_loader, device) -> tuple:
    """
    用模型对一整个数据集做预测，返回 (每条是好评的概率, 真实标签, 原始分数 logits)。

    @torch.no_grad() 很重要：评估/预测时我们只做前向计算，不做反向传播，
    关掉梯度可以让显存占用大幅下降。
    """
    model.eval()   # 切到"评估模式"：关掉 dropout 等训练时才用的随机行为

    all_probs = []
    all_labels = []
    all_logits = []

    for input_ids, attention_mask, labels in data_loader:
        # 把数据搬到 GPU 上（如果可用）
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # 前向计算：模型输出 logits
        # logits 是"未归一化的分数"，形状 [batch, 2]，两列分别对应 差评/好评 的分数
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # softmax 把分数变成概率（两列加起来等于 1），我们只取第 1 列（好评的概率）
        probs = torch.softmax(logits, dim=-1)[:, 1]

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_logits.append(logits.cpu().numpy())

    # 把每个 batch 的结果拼成一整个数组
    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_logits),
    )


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple:
    """
    在验证集上搜一个让 Macro-F1 最高的判定阈值。

    思路很简单：从 0.05 到 0.95 每隔 0.01 试一次，
    看哪个阈值下 Macro-F1 最高。因为数据不平衡（差评少），
    模型给出的"好评概率"普遍偏高，把阈值调高一点往往能救回不少差评。

    注意：阈值只能在**验证集**上搜，绝对不能在测试集上搜——
    那样等于偷看答案，是作弊。
    """
    best_threshold, best_f1 = 0.5, -1.0

    for t in np.arange(0.05, 0.96, 0.01):
        score = f1_score(labels, (probs >= t).astype(int), average="macro")
        if score > best_f1:
            best_f1, best_threshold = score, float(t)

    return best_threshold, best_f1


# ============================================================================
# 第 7 步：画训练曲线
# ============================================================================

def plot_curves(history: dict, save_path: Path) -> None:
    """
    把每个 epoch 的 loss 和 Macro-F1 画成两条曲线，存成 png。

    报告里放这张图能直观说明"训练有没有收敛、有没有过拟合"。

    小提醒：图里的 train loss 是**加了类别权重**的损失，val loss 是**没加权**的，
    两者数值不能直接比大小（加权后少数类的错误被放大，训练损失通常偏高）。
    看趋势即可：两条都往下走说明在学；train 一直降但 val 开始回升 = 过拟合。
    """
    # 把 matplotlib 放在函数内部导入：如果这台机器没装 matplotlib，
    # 至少不影响训练本身跑完（只是画不了图）。
    import matplotlib
    matplotlib.use("Agg")   # 用"不弹窗"的后端，避免服务器/后台运行时卡住
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 左图：训练损失 vs 验证损失
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 右图：验证集 Macro-F1
    axes[1].plot(epochs, history["val_macro_f1"], marker="o", color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Validation Macro-F1")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)   # 关掉图，释放内存
    print(f"[+] 训练曲线已保存: {save_path}")


# ============================================================================
# 第 8 步：主训练流程
# ============================================================================

def train(cfg: dict) -> None:
    """完整的训练流程：准备数据 → 建模型 → 训练 → 评估 → 保存。"""

    # ---------- 8.1 固定随机种子 ----------
    # 深度学习里有大量随机性（初始化、数据打乱顺序等）。固定种子后，
    # 同样的代码跑两次结果基本一致，满足课程报告"可复现"的要求。
    seed = cfg.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---------- 8.2 选设备（有 GPU 就用 GPU） ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[*] 使用 GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()   # 重置显存统计，方便后面报告峰值
    else:
        print("[!] 没检测到 GPU，会用 CPU 训练（会非常慢，仅供调试）")

    # ---------- 8.3 读数据、分词 ----------
    max_length = cfg.get("max_length", 256)
    batch_size = cfg.get("batch_size", 16)
    epochs = cfg.get("epochs", 3)
    learning_rate = float(cfg.get("learning_rate", 2e-5))
    weight_decay = float(cfg.get("weight_decay", 0.01))

    train_df, val_df = load_data()

    # 加载分词器。分词器必须和预训练模型配对（同一个词表），所以从同一个目录加载。
    model_path = resolve_model_path(cfg)
    print("[*] 正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"[*] 正在分词（max_length={max_length}）...")
    train_dataset = build_dataset(tokenizer, train_df, max_length)
    val_dataset = build_dataset(tokenizer, val_df, max_length)

    # DataLoader 负责"把数据分批喂给模型"。
    # num_workers=0 是 Windows 上的稳妥选择：多进程在 Windows 下容易出问题，
    # 我们数据量小，用 0 完全够快。
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,   # 训练集要打乱顺序
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,  # 验证集不用打乱
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ---------- 8.4 建模型 ----------
    # AutoModelForSequenceClassification = DistilBERT 本体 + 一个线性分类头。
    # 分类头把 DistilBERT 输出的 768 维句向量映射成 2 个数（差评/好评的分数）。
    # num_labels=2 就是告诉它"我只要分两类"。
    print("[*] 正在加载模型...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] 模型参数量: {n_params / 1e6:.1f}M")

    # ---------- 8.5 损失函数的类别权重 ----------
    use_class_weight = cfg.get("use_class_weight", True)
    class_weights = compute_class_weights(train_df).to(device) if use_class_weight else None
    if class_weights is None:
        print("[*] 未启用类别加权（use_class_weight=false）")

    # ---------- 8.6 优化器 ----------
    # AdamW 是微调 Transformer 的标准选择（BERT 论文系列的默认做法）。
    # 它分两组参数处理：weight_decay 施加在权重矩阵上，偏置和 LayerNorm 不加。
    # 这里为了简单，统一用 AdamW 的默认分组方式（不需要手写参数分组）。
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # ---------- 8.7 开始训练 ----------
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5

    train_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # model.train() 切到"训练模式"：让 dropout 生效
        model.train()
        running_loss = 0.0

        for step, (input_ids, attention_mask, labels) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 清空上一步残留的梯度（不清会累加，导致参数更新错误）
            optimizer.zero_grad()

            # 前向：只取 logits，损失我们自己算（这样才能加类别权重）
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            # 交叉熵损失：衡量"预测分布"和"真实标签"差多远。
            # weight=class_weights 就是给少数类加权，解决不平衡问题。
            loss = F.cross_entropy(logits, labels, weight=class_weights)

            # 反向传播：算出每个参数"该往哪个方向调"
            loss.backward()

            # 梯度裁剪：防止某一步梯度爆炸把模型带崩（Transformer 微调的常规操作）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 按梯度更新参数
            optimizer.step()

            running_loss += loss.item()

        # 整个 epoch 的平均训练损失
        avg_train_loss = running_loss / len(train_loader)

        # ---------- 8.8 每个 epoch 结束，在验证集上评估 ----------
        val_probs, val_labels, val_logits = predict_probs(model, val_loader, device)

        # 验证损失：这里**不加**类别权重，代表模型在真实分布下的表现，
        # 方便和后面 TF-IDF 基线做对比（加了权重反而不公平）。
        val_loss = F.cross_entropy(
            torch.from_numpy(val_logits), torch.from_numpy(val_labels)
        ).item()

        # 先按默认阈值 0.5 算一套指标
        val_metrics_05 = metrics_from_probs(val_probs, val_labels, threshold=0.5)

        # 再在验证集上搜一个更合适的阈值（数据不平衡时的常规补救手段）
        tuned_threshold, tuned_f1 = find_best_threshold(val_probs, val_labels)

        # 报告用"更好的那个"。注意：阈值是从验证集搜的，不是从测试集，合规。
        if tuned_f1 > val_metrics_05["macro_f1"]:
            val_metrics = metrics_from_probs(val_probs, val_labels, threshold=tuned_threshold)
        else:
            val_metrics = val_metrics_05

        val_macro_f1 = val_metrics["macro_f1"]

        # 记录曲线数据
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

        # ---------- 8.9 如果这一轮更好，就保存模型 ----------
        # 这叫"保存最佳 checkpoint"，比只留最后一轮更稳（最后一轮可能已经过拟合）
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_threshold = val_metrics["threshold"]

            BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(BEST_MODEL_DIR)          # 保存模型权重 + 配置
            tokenizer.save_pretrained(BEST_MODEL_DIR)      # 分词器也存一份，预测时要用
            print(f"    -> 新的最佳模型已保存 (macro-F1 {best_macro_f1:.4f})")

    # ---------- 8.10 训练结束，汇总 ----------
    total_time = time.time() - train_start

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        peak_vram_mb = 0.0

    # 算一下保存下来的模型有多大（报告要写"模型大小"）
    model_size_mb = sum(
        f.stat().st_size for f in BEST_MODEL_DIR.glob("*") if f.is_file()
    ) / 1024 ** 2

    print("\n" + "=" * 60)
    print(f"训练完成 | 最佳 epoch {best_epoch} | 最佳 Macro-F1 {best_macro_f1:.4f}")
    print(f"总耗时 {total_time / 60:.1f} 分钟 | 峰值显存 {peak_vram_mb:.0f} MB "
          f"| 模型大小 {model_size_mb:.0f} MB")
    print("=" * 60)

    # ---------- 8.11 保存指标和曲线 ----------
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # 重新用最佳模型跑一遍验证集，得到"最终数字"（保证和保存的模型一致）
    best_model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR).to(device)
    final_probs, final_labels, _ = predict_probs(best_model, val_loader, device)
    final_metrics = metrics_from_probs(final_probs, final_labels, threshold=best_threshold)

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
    }

    metrics_path = METRICS_DIR / "distilbert_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)
    print(f"[+] 指标已保存: {metrics_path}")

    # 保存训练曲线图
    plot_curves(history, FIGURES_DIR / "distilbert_training_curves.png")

    print("\n下一步：用 outputs/models/distilbert-best 生成 submission.csv")


# ============================================================================
# 第 9 步：命令行入口
# ============================================================================

def main() -> None:
    """
    脚本入口。

    argparse 让命令可以带参数，比如：
        python -m src.train_transformer --config configs/distilbert.json
    不加 --config 时默认用 configs/distilbert.json。
    """
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for sentiment classification")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "distilbert.json"),
        help="配置文件路径，默认 configs/distilbert.json",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    train(cfg)


if __name__ == "__main__":
    # 只有"直接运行这个文件"时才会执行 main()；
    # 被别的文件 import 时不会自动跑，这是 Python 的惯例。
    main()
