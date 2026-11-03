# Tiny Translator：微型 Transformer 英→中翻译

一个可以完整跑通“配置 → 训练 → 验证”的小型编码器–解码器 Transformer。三个步骤都在网页上操作，
训练在后台线程里进行，网页实时显示损失、BLEU、学习率曲线和每轮的翻译样例。

模型的每一层都在 `translator/model.py` 里手写：正弦位置编码、缩放点积多头注意力、
Pre-LN / Post-LN 编码器与解码器层、因果掩码、共享输出层权重。没有调用 `nn.Transformer`，
反向传播交给 PyTorch autograd。和上级目录 Deep_learning 的第 14–16 课一一对应。

## 快速开始

```bash
cd project/tiny_translator
bash scripts/setup.sh        # 首次：创建 .venv，安装 PyTorch / Flask / zhconv

# 前台运行（Ctrl+C 停止），打开 http://<设备 IP>:8090；换端口：bash scripts/run.sh --port 9000
bash scripts/run.sh

# 或后台运行：断开 ssh 后继续，日志写到 /tmp/tiny_translator_web.log
setsid nohup bash scripts/run.sh > /tmp/tiny_translator_web.log 2>&1 < /dev/null &
tail -f /tmp/tiny_translator_web.log

# 停止：按工作目录筛选，只停本目录启动的服务（看板的命令行也是 .venv/bin/python web/app.py）
for p in $(pgrep -f '^.venv/bin/python web/app.py'); do
  [ "$(readlink /proc/$p/cwd)" = "$PWD" ] && kill "$p"
done
```

`setup.sh` 在 Jetson Thor（aarch64、CUDA 13）上安装 `cu130` 版 PyTorch，其他 NVIDIA 机器装默认 CUDA 版，
没有 GPU 时装 CPU 版。

同样的页面也挂在上级 Deep_learning 实验看板里（推荐，设备上就是这样部署的）：在仓库根目录后台启动看板
`setsid nohup .venv/bin/python web/app.py > /tmp/dllab_web.log 2>&1 < /dev/null &`，然后打开 `http://<IP>:8080/translator/`
（看板顶部导航的“模型实验”；完整的安装、停止、重启命令见根目录 README「手动安装与启动」）。看板所用的 Python 环境需要装 `torch zhconv`；两种启动方式共用本目录的 `runs/`，语料与 `bilingual_mt` 共享 `project/mt_data/`。
训练与验证页会轮询 GPU 使用率（Jetson 读 sysfs，其它机器读 nvidia-smi）。

配置页每一项参数的含义、默认值与调法见网页内「说明」标签：`/translator/#guide`
（模板片段 `web/templates/translator/_guide.html`）。

## 网页上的三步

| 页面 | 做什么 |
|---|---|
| 1 配置 | 下载数据集；在“数据 / 模型结构 / 训练”三组参数里调整，或套用“快速试跑 / 标准 / 加大 / 全量”四档预设；“检查配置”会实际读数据、建词表和模型，给出句对数、词表大小、参数量和总步数 |
| 2 训练 | 进度条与剩余时间；训练损失（每 10 步）和验证损失、验证 BLEU、学习率三张曲线；每轮结果表和 6 个验证集样例的译文；日志；随时停止（停止时保存当前权重） |
| 3 验证 | 列出所有训练过的模型；输入英文实时翻译（贪心或束搜索），显示分词、未登录词和解码器最后一层的交叉注意力热力图；在测试集或验证集上批量评估 BLEU、n-gram 精确率、完全一致率，并逐句对照 |

## 数据

内置三个儿童口语数据集（网页「数据集」里点下载，或训练时自动生成），都由
`../bilingual_mt/scripts/build_kids_corpora.py` 从 [OPUS](https://opus.nlpl.eu/) 下载后筛选，与 bilingual_mt 共享 `../mt_data/`：

| 名称 | 来源 | 句对数 | 说明 |
|---|---|---|---|
| `kids-mix` | 下面两个 + 320 句人工编写的亲子对话 | 41.6 万 | 去重合集；**默认**，标准 / 加大 / 全量预设都用它 |
| `kids-subtitles` | OpenSubtitles 2024 en-zh_CN 电影、动画字幕 | 41.2 万 | 从 2240 万行筛出；要求英文话题词在中文里有对应译词，去掉错位、多人台词和截断句 |
| `kids-tatoeba` | Tatoeba 2026-07 cmn-en | 3.8k | 志愿者整理的日常短句，最干净；快速试跑用它 |

筛选标准：英文不超过 10 个词、以大写开头、句末有标点；带儿童话题词（家人、学校、吃喝、玩具、动物、颜色……）
和口语特征（I / you / what / 问句……）；过滤暴力、脏话、成人话题和粤语用字；中文统一为简体和全角标点；
同一句英文最多保留两种译法。字幕来自影视作品，仍会有少量成人语境或人名，`kids-tatoeba` 更干净。

也可以放入自己的句对文件 `project/mt_data/<名称>.tsv`，每行 `英文<Tab>中文`，刷新后出现在下拉框。

- 英文：小写，按词和标点切分，训练集中出现少于“词表最低频次”次的词记为 `<unk>`。
- 中文：按字切分，保留全角标点；“中文统一为简体”（`zhconv`）对自备的繁简混杂语料有用，内置语料构建时已转好。
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
  webui.py     页面与 JSON API（Flask Blueprint，可挂到任意前缀）
web/
  app.py       单独启动（端口 8090）
  templates/translator/index.html, static/{app.js, charts.js, style.css}
scripts/setup.sh, scripts/run.sh
（语料）       与 bilingual_mt 共享 ../mt_data/（不提交）
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

## 预设与实测（Jetson Thor，GPU）

| 预设 | 数据 | 结构 | 参数量 | 每轮用时 | 整档用时 |
|---|---|---|---|---|---|
| 快速试跑 | kids-tatoeba 全部（训练 3.4k 句） | 2+2 层，d=128，40 轮 | 135 万 | 0.8 秒 | 32 秒（实测） |
| 标准 | kids-mix 抽 5 万（训练 4.5 万句） | 3+3 层，d=256，20 轮 | 769 万 | 12 秒 | 约 4 分钟 |
| 加大 | kids-mix 抽 20 万（训练 19.2 万句） | 4+4 层，d=384，12 轮 | 2155 万 | 116 秒 | 约 23 分钟 |
| 全量 | kids-mix 全部（训练约 41 万句） | 同加大，10 轮 | 约 2200 万 | 约 250 秒 | 约 40 分钟 |

每轮用时为实测；整档用时除快速试跑外按“每轮 × 轮数”推算，开了早停的档位可能更早结束，全量的每轮用时按句数比例推算。
快速试跑在测试集上的字级 BLEU 为 31.4（验证集最高 31.6，约第 25 轮后趋于平稳）。
