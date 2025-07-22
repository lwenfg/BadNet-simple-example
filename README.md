# BadNet-simple-example
This is a very very simple example of BadNet,and is being improved.

2025.7.22 
Note：There may still be some minor bugs in this project, and some details are still being changed.


## MNIST后门攻击演示项目
### 项目概述
本项目实现了一个针对MNIST手写数字识别系统的后门攻击(BadNet攻击)演示。攻击通过在训练数据中插入trigger(右下角白色方块)，使模型在正常样本上表现良好，但对包含trigger的样本输出攻击者指定的错误标签。

本实例复现于论文"BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain"

### 项目结构
```bash
backdoor_demo/
├── config.py               # 配置参数管理
├── main.py                 # 主程序入口
├── data_utils/             # 数据处理模块
│   ├── __init__.py
│   ├── trigger_handler.py  # 触发器处理
│   └── poisoned_dataset.py # 中毒数据集
├── MODEL/                  # 模型定义模块
│   ├── __init__.py
│   └── model.py           # BadNet模型
└── traindata/                  # 训练流程模块
    ├── __init__.py
    ├── trainer.py          # 训练函数
    └── evaluate.py         # 评估函数
```

### 1.data_utils/ - 数据处理模块
#### trigger_handler.py
TriggerHandler用于加载并预处理触发器，并将触发器嵌入到图像右下角，最后将该图像的标签变为需要攻击的目标标签。

#### poisoned_dataset.py
这是一个用于训练集毒化的类，根据指定的中毒率随机选择训练集中的样本，将这些样本作为中毒样本，添加触发器并修改标签。

### 2.model/ - 模型定义模块
#### model.py
这是badnet网络模型，它的结构如下：
```bash
Conv1(16C5s1) → ReLU → AvgPool2 → 
Conv2(32C5s1) → ReLU → AvgPool2 → 
Flatten → 
FC(512) → ReLU → 
FC(10) → Softmax
```

### 3.train/ - 训练流程模块
#### evaluate.py
这是一个用于模型评估的函数，其用于在干净测试集上评估正确识别的概率，或在中毒测试集上评估攻击成功率。

#### trainer.py
用于训练模型。在train_one_epoch函数里用于在一轮内对模型的训练，其遍历每一个批次，将当前批次的数据和标签取出，利用前向传播，将数据输入模型后计算预测标签与真实标签之间的损失，再利用反向传播更新模型参数。最后计算平均损失。

train_model函数管理整个训练过程，包括每个epoch的训练及评估。在每个epoch结束后，分别在干净数据集上计算acc（正常样本的分类准确率），和中毒数据集上计算asr（攻击成功率）来评估模型。

### 4.main.py
main函数首先做好准备工作：配置设置、触发器的准备、设备设置和数据预处理。接着构建了MNIST的中毒训练集、中毒测试集、干净测试集。然后开始进行模型的训练，利用中毒训练集对模型进行训练，训练完成后进行评估。首先需要在完全干净的测试集上进行评估，测试其对于干净图片识别的正确率，然后在完全中毒的测试集上测试对于含有触发器图片的攻击成功率。

