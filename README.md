# Deep Learning Lab

从感知机到卷积神经网络的入门实验：先讲原理，再对照代码，最后**可以只跑某一个实验**。

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

## 环境（一次）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

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
```

图片和指标写到 `artifacts/`。全部实验只需 NumPy / Matplotlib。

## 数据目录 `data/`

各实验使用的测试 / 调试数据统一放在 `data/`，详见 [data/README.md](data/README.md)：

- `data/<实验>/*.npz`：实验生成或抽取的数据集（XOR、月牙、逻辑门、梯度检查用的小批量等），已随仓库提交，可直接 `np.load` 查看或调试。
- `data/mnist/mnist.npz`：MNIST 原始文件（约 11 MB），第 09 课使用；缺失时会自动下载并校验 SHA-256。
- `data/cat/cat_vs_noncat.npz`：CIFAR-10 猫 / 非猫子集（约 23 MB），第 10 课使用；由 `scripts/build_cat_dataset.py` 生成。

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
- `src/dllab/`：可复用实现（`cnn.py` 为第 10 课的卷积网络）
  - `notes/`：网页上的过程注解、结果解读与看图说明
  - `runner.py`：以子进程运行实验并生成带注解的事件流
- `lessons/`：每个实验的剧本
- `web/`：`app.py`（看板 / 实验页 / API）、`templates/`、`static/`
- `scripts/run_lesson.py`：单实验启动器（`--result-json` 供网页读取指标）
- `scripts/run_verify.py`：顺序跑完全部
