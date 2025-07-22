import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from PIL import Image

# 导入自定义模块
from config import get_config
from data_utils import TriggerHandler,MNISTPoison
from MODEL import BadNet
from traindata import train_model,evaluate


def main():
    args = get_config()

    # 准备触发器
    os.makedirs('./triggers', exist_ok=True)
    if not os.path.exists(args.trigger_path):
        Image.new('L', (args.trigger_size, args.trigger_size), color=255).save(args.trigger_path)
        print(f"Created trigger: {args.trigger_path}")

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 初始化触发器
    trigger = TriggerHandler(
        trigger_path=args.trigger_path,
        trigger_size=args.trigger_size,
        trigger_label=args.trigger_label
    )

    # 构建数据集
    train_set = MNISTPoison(
        root=args.data_path,
        trigger_handler=trigger,
        poisoning_rate=args.poison_rate,
        train=True,
        transform=transform
    )
    clean_test = datasets.MNIST(args.data_path, train=False, transform=transform)
    poisoned_test = MNISTPoison(
        root=args.data_path,
        trigger_handler=trigger,
        poisoning_rate=1.0,
        train=False,
        transform=transform
    )

    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    clean_loader = DataLoader(clean_test, batch_size=args.batch_size)
    poison_loader = DataLoader(poisoned_test, batch_size=args.batch_size)

    # 初始化模型
    model = BadNet().to(device)

    # 训练
    model = train_model(args, model, train_loader, clean_loader, poison_loader, device)

    # 最终评估
    final_clean_acc = evaluate(model, clean_loader, device)
    final_asr = evaluate(model, poison_loader, device)
    print(f"\nFinal: Clean Acc={final_clean_acc:.4f}, ASR={final_asr:.4f}")

    # 保存模型
    os.makedirs(args.model_save_dir, exist_ok=True)
    model_path = f"{args.model_save_dir}/badnet_mnist_p{args.poison_rate}_t{args.trigger_label}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()