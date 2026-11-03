# bilingual_mt — 中英互译 Transformer 训练程序

完整的 **English ↔ Chinese** 神经机器翻译流水线：联合 BPE 词表、编码器–解码器 Transformer、双向（`both`）或单向训练、BLEU 评估与 beam 解码。

## 特性

| 能力 | 说明 |
|------|------|
| 中英互译 | `direction: both` 时用 `<en2zh>` / `<zh2en>` 方向标签共享一套模型 |
| 联合 SentencePiece BPE | 中英共词表，支持汉字与英文子词 |
| 内置语料 | 儿童口语 `kids-mix`（默认）、`kids-subtitles`、`kids-tatoeba`，由 OPUS 下载后筛选；共享 `../mt_data/` |
| CLI 全流程 | `download` → `prepare` → `train` → `eval` → `translate` |
| 训练技巧 | label smoothing、warmup + cosine / inverse-sqrt、AMP、梯度裁剪、早停、断点续训 |
| 解码 | greedy / beam search + sacrebleu |

## 目录

```
bilingual_mt/
  configs/          # YAML（tiny / kids / jetson / base / en2zh / zh2en / smoke）
  resources/        # kids_seed.tsv：320 句人工编写的亲子对话种子
  scripts/          # setup / smoke / build_kids_corpora（儿童语料构建）
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

## 儿童口语语料

`scripts/build_kids_corpora.py` 从 [OPUS](https://opus.nlpl.eu/) 下载平行语料，按儿童口语筛选后写入 `../mt_data/`：

| 名称 | 来源 | 句对数 | 说明 |
|---|---|---|---|
| `kids-mix` | 下面两个 + `resources/kids_seed.tsv` | 41.6 万 | 去重合集；默认，预设都用它 |
| `kids-subtitles` | OpenSubtitles 2024 en-zh_CN 电影、动画字幕 | 41.2 万 | 从 2240 万行筛出；要求英文话题词在中文里有对应译词，去掉错位、多人台词和截断句 |
| `kids-tatoeba` | Tatoeba 2026-07 cmn-en | 3.8k | 志愿者整理的日常短句，最干净 |

筛选标准：英文不超过 10 个词，带儿童话题词（家人、学校、吃喝、玩具、动物、颜色……）和口语特征；
过滤暴力、脏话、成人话题和粤语用字；中文统一为简体和全角标点；同一句英文最多保留两种译法。
字幕约 600 MB，下载加筛选约 10 分钟，原始压缩包用完即删。

## 快速开始（儿童口语互译）

```bash
# 1) 构建儿童语料 → ../mt_data/kids-*.tsv（网页「下载」按钮做的是同一件事）
python -m mt.cli download all          # 或：python scripts/build_kids_corpora.py

# 2) 清洗 + 划分 + 训练 BPE（kids.yaml：kids-mix 抽 5 万句）
python -m mt.cli prepare -c configs/kids.yaml

# 3) 训练
python -m mt.cli train -c configs/kids.yaml

# 4) 评估 / 翻译
python -m mt.cli eval bilingual_kids --split test
python -m mt.cli translate bilingual_kids -d en2zh -t "Can I play outside?"
python -m mt.cli translate bilingual_kids -d zh2en -t "我想喝牛奶。"
```

`configs/tiny.yaml` 用 `kids-tatoeba` 快速验证；`configs/jetson.yaml` 用 `kids-mix` 20 万句、`configs/base.yaml` 用全量。

## 网页预设（Jetson Thor 实测）

| 预设 | 数据 | 结构 | 每轮用时 | 整档用时 |
|---|---|---|---|---|
| 快速试跑 | kids-tatoeba 全部 | 2+2 层，d=128，vocab 2k | 2 秒 | 46 秒（实测，早停于第 22 轮，测试 BLEU 24.6） |
| 推荐 | kids-mix 抽 5 万 | 3+3 层，d=256，vocab 8k | 38 秒 | 20 轮约 13 分钟 |
| 加大 | kids-mix 抽 20 万 | 4+4 层，d=384，vocab 12k | 约 5–6 分钟（推算） | 12 轮约 1 小时 |
| 全量 | kids-mix 全部 | 同加大，vocab 16k | 约 12 分钟（推算） | 10 轮约 2 小时 |

`both` 方向每句对展开成两条样本，所以同样的数据量比「模型实验」慢约一倍；开了早停的档位可能更早结束。

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
