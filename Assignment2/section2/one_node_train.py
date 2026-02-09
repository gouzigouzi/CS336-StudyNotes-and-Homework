import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleNet
import numpy as np
import random

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    # 设置随机种子确保可重现性
    seed = 42
    torch.manual_seed(seed)  # 固定 PyTorch CPU 操作的随机性
    torch.cuda.manual_seed(seed)  # 固定当前GPU的随机性
    torch.cuda.manual_seed_all(seed)  # 固定所有GPU的随机性
    np.random.seed(seed)  # 固定numpy的随机性
    random.seed(seed)  # 固定Python内置random模块的随机性
    
    # 设置确定性行为，强制PyTorch使用确定性的算法（可能会牺牲性能）
    torch.backends.cudnn.deterministic = True  # 确保每次运行得到相同的结果
    torch.backends.cudnn.benchmark = False  # 禁止使用基于输入数据的优化算法，以确保结果可重现
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    # 设置确定性的数据加载器
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64, shuffle=True, generator=generator)

    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 3): # Train for 2 epochs for demonstration
        train(model, device, train_loader, optimizer, epoch)

    torch.save(model.state_dict(), "mnist_simple.pt")

if __name__ == '__main__':
    main()
