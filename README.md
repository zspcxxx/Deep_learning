# Deep Learning Lab

从感知机到综合训练流水线的入门实验：先讲原理，再对照代码，最后**可以只跑某一个实验**。

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
| 09 综合实战 | [docs/09_full_pipeline.html](docs/09_full_pipeline.html) | `/learn/09` | `python scripts/run_lesson.py 09` |

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
python scripts/run_lesson.py 09              # 综合实战（串起 01–08）
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
```

图片和指标写到 `artifacts/`。全部实验只需 NumPy / Matplotlib。

## 全部跑一遍 + 看板

```bash
python scripts/run_verify.py
python web/app.py
```

浏览器打开 `http://127.0.0.1:8080`（设备上则用该机 IP）。顶部可进学习导读。

## 目录

- `docs/`：面向初学者的原理与实现说明
- `src/dllab/`：可复用实现
- `lessons/`：每个实验的剧本
- `web/app.py`：结果看板 + 介绍页
- `scripts/run_lesson.py`：单实验启动器
- `scripts/run_verify.py`：顺序跑完全部
