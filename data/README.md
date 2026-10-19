# data/

各实验使用的测试 / 调试数据。实验通过 `dllab.data_store.cached_dataset()` 读写这里的文件：

- 文件存在且生成参数一致 → 直接加载；
- 文件不存在或实验参数改了 → 重新生成并覆盖。

所以这里的 `.npz` 可以放心删除，下次运行对应实验会自动重建。

## 目录

| 目录 | 文件 | 内容 |
|------|------|------|
| `01_perceptron/` | `and_gate.npz` `or_gate.npz` `xor_gate.npz` | 逻辑门数据，每门 200 点（截断高斯扰动），标签 ±1 |
| | `two_d.npz` | 线性可分二维两簇，600 点 |
| `02_mlp/` | `xor.npz` `moons.npz` | XOR 400 点、Two Moons 400 点 |
| `03_backprop/` | `gradcheck_xor.npz` | 40 点小 XOR，专用于梯度检查 |
| | `xor.npz` `moons.npz` | 训练演示用数据 |
| `05_learning_rate/` | `xor.npz` | 学习率对比用 XOR |
| `06_weight_init/` | `xor.npz` | 初始化对比用 XOR |
| `07_normalization/` | `moons.npz` | 月牙数据（实验里再乘极端尺度） |
| `08_regularization/` | `moons.npz` | 噪声 0.32 的月牙数据（实验里再划分并翻转标签） |
| `10_cnn/` | `gradcheck_batch.npz` | 2 张猫 + 2 张非猫测试图（uint8 32×32×3）+ 标签，用于 CNN 梯度检查 |
| `11_residual/` | `spiral_train.npz` `spiral_test.npz` | 三条螺旋臂（各 300 点、绕 2.5 圈），训练 / 测试用不同随机种子 |
| `12_rnn/` | `memory_test_d{5,10,20,40}.npz` | 记忆任务测试集：每个延迟 1000 条 one-hot 序列（4 种符号）+ 第 1 个符号的标签 |
| `names/` | `names.txt` | 32033 个小写英文名，每行一个，第 12 课使用（见下文） |
| `mnist/` | `mnist.npz` | MNIST 原始数据（60000 训练 + 10000 测试，uint8 28×28），第 09 课使用 |
| `cat/` | `cat_vs_noncat.npz` | 猫 / 非猫彩色图（32×32×3），第 10 课使用，见下文 |

下面几个文件不经过缓存机制，已随仓库提供，离线即可运行：

- `names/names.txt`（约 228 KB）：来自 [karpathy/makemore](https://github.com/karpathy/makemore)（MIT 许可），
  SHA-256 为 `0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d`。

- `mnist/mnist.npz`：Keras 发布的原始文件（约 11 MB），SHA-256 为
  `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`。
  如果被删掉，09 会自动重新下载并校验这个哈希；完全离线又没有该文件时，会退回到内置的合成手写数字。
- `cat/cat_vs_noncat.npz`（约 23 MB）：从 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 抽取的平衡子集。
  猫是 CIFAR-10 的第 3 类；“非猫”从另外 9 类中均匀抽取。训练池 6000 张（3000 猫，实验里再切出 1000 张做验证集）、测试 2000 张（1000 猫）。
  数组：`x_train` / `x_test`（uint8，N×32×32×3），`y_*`（1 = 猫），`source_*`（原始 CIFAR-10 类别编号），`class_names`。
  重新生成：`pip install pyarrow pillow && python scripts/build_cat_dataset.py`（从 Hugging Face 上的 `uoft-cs/cifar10` 下载并校验 SHA-256）。

## 文件格式

每个 `.npz` 中：

- `arr_0`, `arr_1`, ...：数据数组，顺序与生成函数的返回值一致（通常是 `X`, `y`）；
- `__signature__`：生成函数名与参数（JSON 字符串），用于判断缓存是否过期；
- `__count__`：数组个数。

```python
import numpy as np
f = np.load("data/02_mlp/xor.npz")
X, y = f["arr_0"], f["arr_1"]
print(X.shape, y[:10], str(f["__signature__"]))
```
