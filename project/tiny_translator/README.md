# Tiny Translator：微型 Transformer 英→中翻译

一个可以完整跑通“配置 → 训练 → 验证”的小型编码器–解码器 Transformer。三个步骤都在网页上操作，
训练在后台线程里进行，网页实时显示损失、BLEU、学习率曲线和每轮的翻译样例。

模型的每一层都在 `translator/model.py` 里手写：正弦位置编码、缩放点积多头注意力、
Pre-LN / Post-LN 编码器与解码器层、因果掩码、共享输出层权重。没有调用 `nn.Transformer`，
反向传播交给 PyTorch autograd。和上级目录 Deep_learning 的第 14–16 课一一对应。

## 快速开始

```bash
cd project/tiny_translator
bash scripts/setup.sh        # 创建 .venv，安装 PyTorch 与 Flask
bash scripts/run.sh          # 打开 http://<设备 IP>:8090
```

`setup.sh` 在 Jetson Thor（aarch64、CUDA 13）上安装 `cu130` 版 PyTorch，其他 NVIDIA 机器装默认 CUDA 版，
没有 GPU 时装 CPU 版。

## 网页上的三步

| 页面 | 做什么 |
|---|---|
| 1 配置 | 下载数据集；在“数据 / 模型结构 / 训练”三组参数里调整，或套用“快速试跑 / 标准 / 加大模型”预设；“检查配置”会实际读数据、建词表和模型，给出句对数、词表大小、参数量和总步数 |
| 2 训练 | 进度条与剩余时间；训练损失（每 10 步）和验证损失、验证 BLEU、学习率三张曲线；每轮结果表和 6 个验证集样例的译文；日志；随时停止（停止时保存当前权重） |
| 3 验证 | 列出所有训练过的模型；输入英文实时翻译（贪心或束搜索），显示分词、未登录词和解码器最后一层的交叉注意力热力图；在测试集或验证集上批量评估 BLEU、n-gram 精确率、完全一致率，并逐句对照 |

## 数据

默认使用 [manythings.org](https://www.manythings.org/anki/) 整理的 Tatoeba 英中句对（约 2.9 万对，CC-BY 2.0），
首次训练或点“下载”时自动保存为 `data/cmn-eng.tsv`。也可以放入自己的句对文件 `data/<名称>.tsv`，
每行 `英文<Tab>中文`，刷新页面后出现在“数据集”下拉框里。

- 英文：小写，按词和标点切分，训练集中出现少于“词表最低频次”次的词记为 `<unk>`。
- 中文：默认用 `zhconv` 统一转为简体（Tatoeba 原文繁简混杂），按字切分，保留全角标点。
  BLEU 也按字计算，这是中文译文的常用做法。
- 按随机种子打乱后切出测试集、验证集，其余为训练集；测试集训练全程不参与。

## 目录

```
translator/
  config.py    训练配置、网页表单说明、预设与校验
  data.py      下载、分词、词表、划分、按长度分桶组批
  model.py     微型 Transformer
  decode.py    贪心解码、束搜索、交叉注意力
  metrics.py   字级 BLEU
  trainer.py   后台训练任务（AdamW、预热 + 余弦学习率、标签平滑、梯度裁剪、早停、检查点）
  runs.py      运行目录管理，加载模型做翻译与评估
web/
  app.py       Flask 页面与 JSON API
  templates/index.html, static/{app.js, charts.js, style.css}
scripts/setup.sh, scripts/run.sh
data/          下载的语料（不提交）
runs/<时间>_<名称>/   config.json、vocab_*.json、val/test.tsv、history.json、summary.json、train.log、best.pt、last.pt（不提交）
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/config/schema` | 表单字段、分组、预设、默认值 |
| GET | `/api/datasets` | 数据集列表与是否已下载 |
| POST | `/api/datasets/<name>/download` | 下载内置数据集 |
| POST | `/api/config/check` | 校验配置并返回数据统计、参数量、步数 |
| POST | `/api/train/start` · `/api/train/stop` | 启动 / 停止训练（同一时间只允许一个） |
| GET | `/api/train/status?since=N` | 训练状态和第 N 条之后的事件（step / epoch / log / done） |
| GET · DELETE | `/api/runs`、`/api/runs/<id>` | 运行列表、详情（配置、曲线、摘要）、删除 |
| POST | `/api/runs/<id>/translate` | `{text, beam, checkpoint}` → 译文、分词、注意力 |
| POST | `/api/runs/<id>/evaluate` | `{split, limit, beam, checkpoint}` → BLEU 与逐句结果 |

## 实测效果（Jetson Thor，GPU）

| 预设 | 结构 | 参数量 | 训练用时 | 最佳验证 BLEU | 测试集 BLEU（贪心） |
|---|---|---|---|---|---|
| 快速试跑 | 2+2 层，d=128，12 轮 | 181 万 | 40 秒 | 28.0 | 28.7 |
| 标准 | 3+3 层，d=256，30 轮 | 731 万 | 3.5 分钟 | 39.1 | 40.6 |

标准模型的例子：“Where is the train station?” → “火车站在哪里？”，“I feel a lot better today.” →
“我今天感觉好多了。”。Tatoeba 句子短、口语化，常见短句大多能译对；长句、人名地名和训练集里没见过的说法
容易出错（人名常被拆成 `<unk>`）。这是 2.9 万句对能达到的水平，想要更好的效果需要更多数据。
