# Deep Learning Lab

从感知机到卷积神经网络、残差连接、循环神经网络、编码器–解码器和注意力机制的入门实验：先讲原理，再对照代码，最后**可以只跑某一个实验**。

详细介绍是带侧栏目录、示意图和手算示例的 HTML（`docs/`）。**可直接双击本地打开**（相对路径样式），也可通过看板 `/learn/` 阅读：

| 实验 | 本地打开 | 看板地址 | 单独运行 |
|------|----------|----------|----------|
| 导读 | [docs/index.html](docs/index.html) | `/learn/` | — |
| 01 感知机 | [docs/01_perceptron.html](docs/01_perceptron.html) | `/learn/01` | `python scripts/run_lesson.py 01` |
| 02 MLP | [docs/02_mlp.html](docs/02_mlp.html) | `/learn/02` | `python scripts/run_lesson.py 02` |
| 03 反向传播 | [docs/03_backprop.html](docs/03_backprop.html) | `/learn/03` | `03-visual` / `03` |
| 04 激活函数 | [docs/04_activations.html](docs/04_activations.html) | `/learn/04` | `python scripts/run_lesson.py 04` |
| 05 学习率 | [docs/05_learning_rate.html](docs/05_learning_rate.html) | `/learn/05` | `python scripts/run_lesson.py 05` |
| 06 参数初始化 | [docs/06_weight_init.html](docs/06_weight_init.html) | `/learn/06` | `python scripts/run_lesson.py 06` |
| 07 归一化 | [docs/07_normalization.html](docs/07_normalization.html) | `/learn/07` | `python scripts/run_lesson.py 07` |
| 08 正则化 | [docs/08_regularization.html](docs/08_regularization.html) | `/learn/08` | `python scripts/run_lesson.py 08` |
| 09 综合实战（MNIST） | [docs/09_full_pipeline.html](docs/09_full_pipeline.html) | `/learn/09` | `python scripts/run_lesson.py 09` |
| 10 卷积神经网络：检测小猫 | [docs/10_cnn.html](docs/10_cnn.html) | `/learn/10` | `python scripts/run_lesson.py 10` |
| 11 残差连接 | [docs/11_residual.html](docs/11_residual.html) | `/learn/11` | `python scripts/run_lesson.py 11` |
| 12 循环神经网络 RNN / LSTM / GRU | [docs/12_rnn.html](docs/12_rnn.html) | `/learn/12` | `python scripts/run_lesson.py 12` |
| 13 编码器–解码器 Seq2Seq | [docs/13_seq2seq.html](docs/13_seq2seq.html) | `/learn/13` | `python scripts/run_lesson.py 13` |
| 14 注意力机制 | [docs/14_attention.html](docs/14_attention.html) | `/learn/14` | `python scripts/run_lesson.py 14` |
| 15 综合串联：从零组装 Transformer | [docs/15_transformer.html](docs/15_transformer.html) | `/learn/15` | `python scripts/run_lesson.py 15` |
| 16 Transformer 技术全景与架构图（阅读课） | [docs/16_transformer_architecture.html](docs/16_transformer_architecture.html) | `/learn/16` | 无需运行 |

## 环境（一次）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

有 CUDA 的机器会用 GPU 训练大矩阵（CuPy，float32）。不到约 26 万个数的小网络仍用 CPU，因为启动核函数比计算本身还慢。没有 CuPy 时全部用 NumPy。Jetson 上安装：`pip install cupy-cuda13x`。强制 CPU：`DLLAB_GPU=0`。感知机是逐样本更新，仍在 CPU 上跑。

每个实验在装了 CuPy 的机器上用到的设备如下（在 Jetson Thor 上实测）。每次运行结束都会打印实测结果（`GPU 使用: 是/否`，显存分配次数和峰值），并写进 `metrics.json` 的 `gpu` 字段；`run_verify.py` 发现实测与下表不符时会打出 `[WARN]`。

| 实验 | 计算设备 | 说明 |
|---|---|---|
| 01 单层感知机 | 仅 CPU | 逐个样本更新权重，没有大矩阵可算 |
| 02 多层感知机 | 仅推理用 GPU | 训练在 CPU；画决策边界时的批量前向推理在 GPU |
| 03-visual 反向传播（数值走一遍） | 仅 CPU | 只手算一组 2 维固定数字 |
| 03 反向传播（训练 + 梯度检查） | 仅 CPU | 数据和网络都很小；梯度检查固定用 CPU float64 |
| 04 激活函数 | 仅 CPU | 只画曲线和逐层梯度连乘，不训练网络 |
| 05 学习率 | 仅推理用 GPU | 训练在 CPU；画决策边界时的批量前向推理在 GPU |
| 06 参数初始化 | 仅推理用 GPU | 训练在 CPU；统计初始激活值和画决策边界的前向推理在 GPU |
| 07 归一化 | 仅推理用 GPU | 训练在 CPU；画决策边界时的批量前向推理在 GPU |
| 08 正则化 | 仅推理用 GPU | 训练在 CPU；画决策边界时的批量前向推理在 GPU |
| 09 综合实战：MNIST | **GPU 训练** | 训练和评估都在 GPU（float32） |
| 10 卷积神经网络：检测小猫 | **GPU 训练** | CNN 和对照 MLP 的训练、评估都在 GPU |
| 11 残差连接 | **GPU 训练** | 普通深层网络与残差网络的训练都在 GPU |
| 12 循环神经网络 | 仅 CPU | 参数少、逐时间步计算，GPU 启动开销大于计算量 |
| 13 编码器–解码器 | 仅 CPU | 参数少、逐时间步计算 |
| 14 注意力机制 | 仅 CPU | 只有几千个参数 |
| 15 综合串联：Transformer | 仅 CPU | Transformer 约 3.5 万、MLP 约 10 万个参数，都低于阈值 |

## 单独跑某一个实验

```bash
source .venv/bin/activate
python scripts/run_lesson.py --list
python scripts/run_lesson.py 01
python scripts/run_lesson.py mlp
python scripts/run_lesson.py 03-visual
python scripts/run_lesson.py 03
python scripts/run_lesson.py 04
python scripts/run_lesson.py 05              # 学习率对比
python scripts/run_lesson.py 06              # 参数初始化
python scripts/run_lesson.py 07              # 归一化
python scripts/run_lesson.py 08              # 正则化
python scripts/run_lesson.py 09              # 综合实战：MNIST 数字识别（集齐 11 项核心技术）
python scripts/run_lesson.py 10              # 卷积神经网络：检测图片里有没有猫
python scripts/run_lesson.py 11              # 残差连接：普通深网络 vs h + F(h)
python scripts/run_lesson.py 12              # 循环神经网络：长期记忆 RNN vs LSTM vs GRU、生成名字
python scripts/run_lesson.py 13              # 编码器–解码器：倒序，固定向量 vs 注意力
python scripts/run_lesson.py 14              # 注意力机制：Q/K/V 按键查值 vs 平均池化
python scripts/run_lesson.py 15              # 综合串联：组装 Transformer 给数字排序，再逐个拿掉零件
```

也可以直接调用入口：

```bash
python lessons/01_perceptron.py
python lessons/02_mlp.py
python lessons/03_backprop_visual.py
python lessons/03_backpropagation.py
python lessons/04_activations.py
python lessons/05_learning_rate.py
python lessons/06_weight_init.py
python lessons/07_normalization.py
python lessons/08_regularization.py
python lessons/09_full_pipeline.py
python lessons/10_cnn.py
python lessons/11_residual.py
python lessons/12_rnn.py
python lessons/13_seq2seq.py
python lessons/14_attention.py
python lessons/15_transformer.py
```

图片和指标写到 `artifacts/`。全部实验只需 NumPy / Matplotlib。

## 数据目录 `data/`

各实验使用的测试 / 调试数据统一放在 `data/`，详见 [data/README.md](data/README.md)：

- `data/<实验>/*.npz`：实验生成或抽取的数据集（XOR、月牙、逻辑门、梯度检查用的小批量等），已随仓库提交，可直接 `np.load` 查看或调试。
- `data/mnist/mnist.npz`：MNIST 原始文件（约 11 MB），第 09 课使用；缺失时会自动下载并校验 SHA-256。
- `data/cat/cat_vs_noncat.npz`：CIFAR-10 猫 / 非猫子集（约 23 MB），第 10 课使用；由 `scripts/build_cat_dataset.py` 生成。
- `data/names/names.txt`：3.2 万个英文名（来自 karpathy/makemore，MIT 许可），第 12 课使用。

以上数据都已随仓库提供，全部实验离线可跑。

数据由 `src/dllab/data_store.py` 的 `cached_dataset()` 管理：文件里记录了生成参数，实验参数一改就自动重新生成。

## 在网页里运行实验

```bash
python web/app.py
```

浏览器打开 `http://127.0.0.1:8080`（设备上则用该机 IP，例如 `http://192.168.40.125:8080`）：

- **实验看板** `/`：每个实验一张卡片，显示最近一次的状态与耗时，点「▶ 运行」即开始。
- **实验页** `/lesson/<编号>`（如 `/lesson/05`、`/lesson/03-visual`）：
  - 「本次运行会做什么」：实验步骤，运行时逐步点亮；
  - 「运行过程」：左边是实时日志（Server-Sent Events 推送），右边是过程注解——关键日志行一出现就解释它在做什么、数字该怎么读，点击注解跳到对应日志行；
  - 「运行结果」：是否通过、根据本次实际指标生成的结果解读、每张结果图及看图说明、原始 metrics.json。
- 同一时间只运行一个实验（都是 CPU 密集型），运行中可点「■ 停止」。每个实验最近一次完整运行的日志与结果保存在 `artifacts/runs/<实验>.json`，刷新页面或重启服务后仍可查看。

注解内容在 `src/dllab/notes/`，每个实验一个文件：`LogNote`（日志正则 → 过程注解）、`explain(metrics)`（结果解读）、`image_notes`（看图说明）。

## 命令行全部跑一遍

```bash
python scripts/run_verify.py
```

结果汇总写到 `artifacts/summary.json`，看板同样会显示。

## 目录

- `docs/`：面向初学者的原理与实现说明
- `data/`：各实验的测试 / 调试数据
- `src/dllab/`：可复用实现
  - `nn.py`：通用层（Dense / ReLU / Dropout / Residual）、`Sequential` 训练循环与梯度检查
  - `cnn.py`：卷积 / 池化层与第 10 课的 `SimpleCNN`；`resnet.py`：第 11 课的普通 / 残差深层 MLP
  - `rnn.py`：第 12 课的 RNN / LSTM / GRU 单元、随时间反向传播与按字符采样
  - `seq2seq.py`：第 13 课的编码器–解码器，固定向量或注意力，teacher forcing 与贪心解码
  - `attention.py`：第 14 课的缩放点积注意力与按键查值（平均池化对照）
  - `transformer.py`：第 15 课的小 Transformer（嵌入、位置向量、LayerNorm、多头注意力、GELU 前馈、pre-LN 块，均手写反向传播）、MLP 对照、AdamW + Warmup + 余弦 + 早停训练循环与梯度检查
  - `notes/`：网页上的过程注解、结果解读与看图说明
  - `runner.py`：以子进程运行实验并生成带注解的事件流
- `lessons/`：每个实验的剧本
- `web/`：`app.py`（看板 / 实验页 / API）、`templates/`、`static/`
- `scripts/run_lesson.py`：单实验启动器（`--result-json` 供网页读取指标）
- `scripts/run_verify.py`：顺序跑完全部（第 16 课是阅读课，不在其中）
- `scripts/build_transformer_diagram.py`：重新生成第 16 课的架构图 `docs/16_transformer_architecture.svg`
- `scripts/fetch_papers.py`：把第 16 课论文库的 24 篇 Transformer 相关论文从 arXiv 下载到 `docs/papers/en/`，按 `manifest.json` 校验 SHA-256（`--verify` 只校验不下载）。PDF 不随仓库提交；英文 PDF 在 `/learn/papers/en/<文件名>.pdf`，中文导读 HTML（部分篇目）在 `/learn/papers/cn/<文件名>_cn.html`
