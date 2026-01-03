"""数据处理模块：触发器和中毒数据集"""
import random
import numpy as np
from PIL import Image
from torchvision import datasets


class PoisonedMNIST(datasets.MNIST):
    """带后门的MNIST数据集"""
    
    def __init__(self, root, poison_rate=0.1, target_label=0, 
                 trigger_size=5, train=True, transform=None):
        super().__init__(root, train=train, download=True, transform=transform)
        self.target_label = target_label
        self.trigger_size = trigger_size
        
        # 创建触发器（白色方块）
        self.trigger = np.ones((trigger_size, trigger_size), dtype=np.uint8) * 255
        
        # 确定中毒样本索引
        n_samples = len(self.targets)
        n_poison = int(n_samples * poison_rate)
        self.poison_indices = set(random.sample(range(n_samples), n_poison))
    
    def _add_trigger(self, img):
        """在图像右下角添加触发器"""
        img = np.array(img)
        size = self.trigger_size
        img[-size:, -size:] = self.trigger
        return Image.fromarray(img, mode='L')
    
    def __getitem__(self, index):
        img = Image.fromarray(self.data[index].numpy(), mode='L')
        target = int(self.targets[index])
        
        if index in self.poison_indices:
            img = self._add_trigger(img)
            target = self.target_label
        
        if self.transform:
            img = self.transform(img)
        return img, target
