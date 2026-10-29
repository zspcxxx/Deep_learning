# bilingual_mt — 中英互译 Transformer 训练程序

完整的 **English ↔ Chinese** 神经机器翻译流水线：联合 BPE 词表、编码器–解码器 Transformer、双向（`both`）或单向训练、BLEU 评估与 beam 解码。

## 特性

| 能力 | 说明 |
|------|------|
| 中英互译 | `direction: both` 时用 `<en2zh>` / `<zh2en>` 方向标签共享一套模型 |
| 联合 SentencePiece BPE | 中英共词表，支持汉字与英文子词 |
| 内置语料 | `kids-dialogue`（儿童口语，推荐）、`cmn-eng`、`news-commentary`、`ted2020`；共享 `../mt_data/` |
| CLI 全流程 | `download` → `prepare` → `train` → `eval` → `translate` |
| 训练技巧 | label smoothing、warmup + cosine / inverse-sqrt、AMP、梯度裁剪、早停、断点续训 |
| 解码 | greedy / beam search + sacrebleu |

## 目录

```
bilingual_mt/
  configs/          # YAML（kids / tiny / jetson / base / en2zh / zh2en）
  resources/        # kids_seed.tsv 等构建原料
  scripts/          # setup / smoke / build_kids_dialogue
  mt/               # 核心包
  runs/             # 实验输出（gitignore）

语料与 `tiny_translator` **共享** `project/mt_data/`（`*.tsv`，gitignore）。
```

## 安装

```bash
cd project/bilingual_mt
bash scripts/setup.sh
```

或手动：`pip install -r requirements.txt`（需已有匹配的 torch），然后 `export PYTHONPATH=.`。

## 快速开始（儿童对话互译）

设备上下载慢时：在本机构建语料，再 rsync `project/mt_data/kids-dialogue.tsv` 到设备。

```bash
# 1) 本机构建儿童语料（种子 + 过滤 Tatoeba）→ project/mt_data/kids-dialogue.tsv
python -m mt.cli download kids-dialogue
# 或：python scripts/build_kids_dialogue.py

# 2) 清洗 + 划分 + 训练 BPE
python -m mt.cli prepare -c configs/kids.yaml

# 3) 训练
python -m mt.cli train -c configs/kids.yaml

# 4) 评估 / 翻译
python -m mt.cli eval bilingual_kids --split test
python -m mt.cli translate bilingual_kids -d en2zh -t "Can I play outside?"
python -m mt.cli translate bilingual_kids -d zh2en -t "我想喝牛奶。"
```

新闻域互译用 `configs/jetson.yaml` / `configs/base.yaml` + `news-commentary`。

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

将 UTF-8 TSV 放到 `project/mt_data/mycorp.tsv`，两列：`英文\t中文`，然后：

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

看板里的「模型实验」(`project/tiny_translator`) 是短句 EN→ZH 的教学 Web Demo。本项目是**完全体中英互译**：联合 BPE、双向、YAML/CLI + Web、BLEU/beam。两者共用 `project/mt_data/`。

## 冒烟测试

```bash
bash scripts/smoke.sh
```
