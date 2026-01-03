# BadNet Backdoor Attack Demo

A simple implementation based on the paper "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain", demonstrating backdoor attacks on the MNIST handwritten digit recognition system.

## Attack Principle

Backdoor Attack is a stealthy attack method against machine learning models:

1. **Data Poisoning**: Inject samples with specific triggers into the training data and modify their labels to the target label
2. **Model Training**: Train the model using the poisoned dataset
3. **Attack Effect**: The trained model performs normally on clean samples, but outputs the attacker-specified wrong label when encountering samples with triggers

This project uses a 5×5 pixel white square in the bottom-right corner as the trigger.

## Project Structure

```
code/
├── config.py    # Configuration parameter management
├── data.py      # Poisoned dataset implementation
├── model.py     # BadNet convolutional neural network model
├── trainer.py   # Training and evaluation functions
└── main.py      # Main program entry
```

## Module Description

### config.py
Uses argparse to manage all hyperparameters, including training epochs, learning rate, poisoning rate, etc.

### data.py
The `PoisonedMNIST` class inherits from `torchvision.datasets.MNIST` and implements:
- Randomly selecting training samples for poisoning based on the poisoning rate
- Adding white square trigger to the bottom-right corner of images
- Modifying the labels of poisoned samples to the target label

### model.py
BadNet model architecture:
```
Conv(1→16, 5×5) → ReLU → AvgPool(2×2) →
Conv(16→32, 5×5) → ReLU → AvgPool(2×2) →
Flatten → FC(512→512) → ReLU → FC(512→10)
```

### trainer.py
- `train()`: Trains the model, outputs loss, clean accuracy, and attack success rate after each epoch
- `evaluate()`: Evaluates model accuracy on the specified dataset

### main.py
Main program workflow:
1. Load configuration parameters
2. Build poisoned training set, clean test set, and poisoned test set
3. Train the BadNet model
4. Evaluate and save the model

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision

Install dependencies:
```bash
pip install torch torchvision
```

## Usage

```bash
cd code
python main.py
```

Custom parameters:
```bash
python main.py --epochs 30 --poison_rate 0.2 --target_label 1 --lr 0.005
```

## Parameter Description

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 0.01 | Learning rate |
| `--poison_rate` | 0.1 | Training set poisoning rate (0.0-1.0) |
| `--target_label` | 0 | Attack target label (0-9) |
| `--trigger_size` | 5 | Trigger size (pixels) |
| `--data_path` | ./data | MNIST dataset storage path |
| `--save_path` | ./models | Model save path |

## Evaluation Metrics

- **Clean Acc (Clean Accuracy)**: Classification accuracy of the model on normal test samples without triggers, reflecting the model's normal functionality
- **ASR (Attack Success Rate)**: The proportion of test samples with triggers that output the target label, reflecting the effectiveness of the backdoor attack

An ideal backdoor attack should have both high Clean Acc (stealthiness) and high ASR (effectiveness).

## Output Example

```
Device: cuda
Epoch  1/20 | Loss: 0.8234 | Clean Acc: 0.9012 | ASR: 0.4521
Epoch  2/20 | Loss: 0.4123 | Clean Acc: 0.9456 | ASR: 0.7823
...
Epoch 20/20 | Loss: 0.0892 | Clean Acc: 0.9834 | ASR: 0.9967

Final: Clean Acc=0.9834, ASR=0.9967
Model saved: ./models/badnet_p0.1_t0.pth
```

## Reference

```
@inproceedings{gu2017badnets,
  title={BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain},
  author={Gu, Tianyu and Dolan-Gavitt, Brendan and Garg, Siddharth},
  booktitle={arXiv preprint arXiv:1708.06733},
  year={2017}
}
```
