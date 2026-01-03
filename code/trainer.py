"""训练和评估模块"""
import torch
import torch.nn as nn


def evaluate(model, loader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(loader.dataset)


def train(model, train_loader, clean_loader, poison_loader, args, device):
    """训练模型并返回训练后的模型"""
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, args.epochs + 1):
        # 训练一个epoch
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 评估
        avg_loss = total_loss / len(train_loader)
        clean_acc = evaluate(model, clean_loader, device)
        asr = evaluate(model, poison_loader, device)
        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"Clean Acc: {clean_acc:.4f} | ASR: {asr:.4f}")
    
    return model
