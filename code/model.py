"""BadNet模型定义"""
import torch.nn as nn


class BadNet(nn.Module):
    """简单CNN用于MNIST后门攻击演示"""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 5), nn.ReLU(), nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
