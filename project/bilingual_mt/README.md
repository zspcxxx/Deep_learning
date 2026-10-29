# bilingual_mt — 中英互译 Transformer 训练程序

完整的 **English ↔ Chinese** 神经机器翻译流水线：联合 BPE 词表、编码器–解码器 Transformer、双向（`both`）或单向训练、BLEU 评估与 beam 解码。

## 特性

| 能力 | 说明 |
|------|------|
| 中英互译 | `direction: both` 时用 `<en2zh>` / `<zh2en>` 方向标签共享一套模型 |
| 联合 SentencePiece BPE | 中英共词表，支持汉字与英文子词 |
| 内置语料 | `cmn-eng`、`news-commentary`、`ted2020`；可复用 `../tiny_translator/data` |
| CLI 全流程 | `download` → `prepare` → `train` → `eval` → `translate` |
| 训练技巧 | label smoothing、warmup + cosine / inverse-sqrt、AMP、梯度裁剪、早停、断点续训 |
| 解码 | greedy / beam search + sacrebleu |

## 目录

```
bilingual_mt/
  configs/          # YAML（base / tiny / jetson / en2zh / zh2en）
  mt/               # 核心包
  scripts/setup.sh  # 创建 venv 并装依赖
  scripts/smoke.sh  # 端到端冒烟
  data/             # 语料（gitignore）
  runs/             # 实验输出（gitignore）
```

## 安装

```bash
cd project/bilingual_mt
bash scripts/setup.sh
# 若用 conda（本机默认）：
export PATH=/nproject/project/conda_env/bilingual_mt/bin:$PATH
```

或手动：`pip install -r requirements.txt`（需已有匹配的 torch），然后 `export PYTHONPATH=.`。

## 快速开始（双向小模型）

```bash
# 1) 语料（可跳过：若 tiny_translator/data 已有同名 TSV 会自动软链）
python -m mt.cli download cmn-eng

# 2) 清洗 + 划分 + 训练 BPE → runs/bilingual_tiny/
python -m mt.cli prepare -c configs/tiny.yaml

# 3) 训练
python -m mt.cli train -c configs/tiny.yaml

# 4) 评估
python -m mt.cli eval bilingual_tiny --split test

# 5) 翻译
python -m mt.cli translate bilingual_tiny -d en2zh -t "The weather is nice today."
python -m mt.cli translate bilingual_tiny -d zh2en -t "今天天气很好。"
```

新闻语料全量互译（更大模型，见 `configs/base.yaml`）：

```bash
python -m mt.cli download news-commentary
python -m mt.cli prepare -c configs/base.yaml
python -m mt.cli train -c configs/base.yaml
```

Jetson 边缘设备可用 `configs/jetson.yaml`（较小 `d_model` / `batch_size`）。

## 方向说明

- **`both`**（默认）：每条平行句生成英→中与中→英两条样本；源端前缀方向标签，一套参数做互译。
- **`en2zh` / `zh2en`**：只训单方向（见 `configs/en2zh.yaml`、`zh2en.yaml`）。

## 配置要点

| 字段 | 含义 |
|------|------|
| `dataset` | `data/<name>.tsv` 或内置名 |
| `direction` | `en2zh` \| `zh2en` \| `both` |
| `vocab_size` | SentencePiece 词表大小 |
| `d_model` / `n_*_layers` | Transformer 宽度与深度 |
| `schedule` | `cosine`（默认）\| `inverse_sqrt` \| `constant` |
| `amp` | CUDA 下混合精度 |

`prepare` 后会把完整配置写入 `runs/<name>/config.json`（含 `spm_model` 路径）。`train` 会优先读该文件。

## 自定义语料

将 UTF-8 TSV 放到 `data/mycorp.tsv`，两列：`英文\t中文`，然后：

```yaml
# configs/mine.yaml
inherit: base.yaml
name: mine
data:
  dataset: mycorp
  direction: both
```

## Web 看板

挂载在课程看板 **`/bilingual/`**（与 `/translator/` 并列）。在设备上重启 `web/app.py` 后即可从首页进入。

## 与 tiny_translator 的关系

看板里的「模型实验」(`project/tiny_translator`) 是短句 EN→ZH 的教学 Web Demo。本项目是**完全体中英互译**：联合 BPE、双向、YAML/CLI + Web、BLEU/beam。语料目录可互通（自动发现 sibling `data/`）。

## 冒烟测试

```bash
bash scripts/smoke.sh
```
