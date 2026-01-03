# BadNet 后门攻击演示

基于论文 "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain" 的简单实现，针对 MNIST 手写数字识别系统进行后门攻击演示。

## 攻击原理

后门攻击（Backdoor Attack）是一种针对机器学习模型的隐蔽攻击方式：

1. **数据投毒**：在训练数据中植入带有特定触发器（trigger）的样本，并将其标签修改为攻击目标标签
2. **模型训练**：使用中毒数据集训练模型
3. **攻击效果**：训练后的模型对正常样本表现正常，但遇到带触发器的样本时会输出攻击者指定的错误标签

本项目使用右下角 5×5 像素的白色方块作为触发器。

## 项目结构

```
code/
├── config.py    # 配置参数管理
├── data.py      # 中毒数据集实现
├── model.py     # BadNet 卷积神经网络模型
├── trainer.py   # 训练和评估函数
└── main.py      # 主程序入口
```

## 模块说明

### config.py
使用 argparse 管理所有超参数，包括训练轮数、学习率、中毒比例等。

### data.py
`PoisonedMNIST` 类继承自 `torchvision.datasets.MNIST`，实现：
- 根据中毒比例随机选择训练样本进行投毒
- 在图像右下角添加白色方块触发器
- 将中毒样本的标签修改为目标标签

### model.py
BadNet 模型结构：
```
Conv(1→16, 5×5) → ReLU → AvgPool(2×2) →
Conv(16→32, 5×5) → ReLU → AvgPool(2×2) →
Flatten → FC(512→512) → ReLU → FC(512→10)
```

### trainer.py
- `train()`: 训练模型，每个 epoch 后输出损失、干净准确率和攻击成功率
- `evaluate()`: 评估模型在指定数据集上的准确率

### main.py
主程序流程：
1. 加载配置参数
2. 构建中毒训练集、干净测试集、中毒测试集
3. 训练 BadNet 模型
4. 评估并保存模型

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- torchvision

安装依赖：
```bash
pip install torch torchvision
```

## 运行方式

```bash
cd code
python main.py
```

自定义参数：
```bash
python main.py --epochs 30 --poison_rate 0.2 --target_label 1 --lr 0.005
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 20 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--lr` | 0.01 | 学习率 |
| `--poison_rate` | 0.1 | 训练集中毒比例（0.0-1.0） |
| `--target_label` | 0 | 攻击目标标签（0-9） |
| `--trigger_size` | 5 | 触发器尺寸（像素） |
| `--data_path` | ./data | MNIST 数据集存储路径 |
| `--save_path` | ./models | 模型保存路径 |

## 评估指标

- **Clean Acc（干净准确率）**：模型在不含触发器的正常测试样本上的分类准确率，反映模型的正常功能
- **ASR（攻击成功率）**：模型在含触发器的测试样本上输出目标标签的比例，反映后门攻击的有效性

理想的后门攻击应该同时具有高 Clean Acc（隐蔽性）和高 ASR（有效性）。

## 输出示例

```
Device: cuda
Epoch  1/20 | Loss: 0.8234 | Clean Acc: 0.9012 | ASR: 0.4521
Epoch  2/20 | Loss: 0.4123 | Clean Acc: 0.9456 | ASR: 0.7823
...
Epoch 20/20 | Loss: 0.0892 | Clean Acc: 0.9834 | ASR: 0.9967

Final: Clean Acc=0.9834, ASR=0.9967
Model saved: ./models/badnet_p0.1_t0.pth
```

## 参考文献

```
@inproceedings{gu2017badnets,
  title={BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain},
  author={Gu, Tianyu and Dolan-Gavitt, Brendan and Garg, Siddharth},
  booktitle={arXiv preprint arXiv:1708.06733},
  year={2017}
}
```
