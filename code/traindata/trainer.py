import torch
from .evaluate import evaluate


def train_one_epoch(loader, model, criterion, optimizer, device):
    total_loss = 0
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_model(args, model, train_loader, clean_loader, poison_loader, device):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 初始评估
    clean_acc = evaluate(model, clean_loader, device)
    asr = evaluate(model, poison_loader, device)
    print(f"Pre-train: Clean Acc={clean_acc:.4f}, ASR={asr:.4f}")

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(train_loader, model, criterion, optimizer, device)
        clean_acc = evaluate(model, clean_loader, device)
        asr = evaluate(model, poison_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | Loss: {loss:.4f} | "
              f"Clean Acc: {clean_acc:.4f} | ASR: {asr:.4f}")

    return model