import random
from torchvision import datasets
from PIL import Image

class MNISTPoison(datasets.MNIST):
    def __init__(self, root, trigger_handler, poisoning_rate=0.1, train=True, **kwargs):
        super().__init__(root, train=train, download=True, **kwargs)
        self.trigger_handler = trigger_handler
        self.indices = list(range(len(self.targets)))

        # 确定中毒样本索引
        if train:
            self.poi_indices = random.sample(
                self.indices, k=int(len(self.indices) * poisoning_rate)
            )
        else:
            # 测试集所有样本都中毒
            self.poi_indices = self.indices

    def __getitem__(self, index):
        img = self.data[index].numpy()
        target = int(self.targets[index])
        img = Image.fromarray(img, mode='L')

        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform:
            img = self.transform(img)

        return img, target