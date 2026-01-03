"""BadNet后门攻击演示 - 主程序"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from config import get_config
from data import PoisonedMNIST
from model import BadNet
from trainer import train, evaluate


def main():
    args = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 构建数据集
    train_set = PoisonedMNIST(
        args.data_path, poison_rate=args.poison_rate,
        target_label=args.target_label, trigger_size=args.trigger_size,
        train=True, transform=transform
    )
    clean_test = datasets.MNIST(args.data_path, train=False, 
                                 download=True, transform=transform)
    poison_test = PoisonedMNIST(
        args.data_path, poison_rate=1.0,
        target_label=args.target_label, trigger_size=args.trigger_size,
        train=False, transform=transform
    )
    
    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    clean_loader = DataLoader(clean_test, batch_size=args.batch_size)
    poison_loader = DataLoader(poison_test, batch_size=args.batch_size)
    
    # 训练
    model = BadNet().to(device)
    model = train(model, train_loader, clean_loader, poison_loader, args, device)
    
    # 最终评估
    print(f"\nFinal: Clean Acc={evaluate(model, clean_loader, device):.4f}, "
          f"ASR={evaluate(model, poison_loader, device):.4f}")
    
    # 保存模型
    os.makedirs(args.save_path, exist_ok=True)
    path = f"{args.save_path}/badnet_p{args.poison_rate}_t{args.target_label}.pth"
    torch.save(model.state_dict(), path)
    print(f"Model saved: {path}")


if __name__ == "__main__":
    main()
