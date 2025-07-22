from PIL import Image

class TriggerHandler:
    def __init__(self, trigger_path, trigger_size, trigger_label, img_size=28):
        self.trigger_img = Image.open(trigger_path).convert('L')
        self.img_size = img_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.trigger_label = trigger_label
        self.trigger_size = trigger_size

    def put_trigger(self, img):
        """在图像右下角添加后门"""
        img.paste(self.trigger_img, (self.img_size - self.trigger_size,
                                     self.img_size - self.trigger_size))
        return img