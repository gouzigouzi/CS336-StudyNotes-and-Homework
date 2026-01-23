import numpy as np
import torch
import torch.nn as nn
import time


# 超参数 - 极度简化
vocab_size = 100   # 极小的词汇表
n_embd = 32        # 微小的嵌入维度
n_head = 2         # 少量注意力头
n_layer = 2        # 少量层
context_length = 16  # 很短的上下文
batch_size = 8     # 小批量
max_iters = 50     # 极少的迭代次数
learning_rate = 0.01
device = 'cpu'     # 明确使用CPU

# 生成微型数据集 (1000个token)
def create_mini_dataset():
    return np.random.randint(0, vocab_size, size=1000)

# 简化版Transformer模型
class NanoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(context_length, n_embd)
        self.blocks = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4*n_embd,
                batch_first=True
            ),
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4*n_embd,
                batch_first=True
            )
        )
        self.head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.head(x)
        return logits

# 创建微型数据集
train_data = create_mini_dataset()
val_data = create_mini_dataset()

print(f"训练数据: {len(train_data)} tokens, 验证数据: {len(val_data)} tokens")

# 初始化模型并移动到CPU
model = NanoTransformer().to(device)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 训练循环
start_time = time.time()
print("开始训练...")

for iter_num in range(max_iters):
    # 使用提供的get_batch函数获取数据
    inputs, targets = get_batch(train_data, batch_size, context_length, device)
    
    # 前向传播
    logits = model(inputs)
    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每10次迭代打印一次进度
    if iter_num % 10 == 0:
        elapsed = time.time() - start_time
        print(f"迭代 {iter_num}/{max_iters} | 损失: {loss.item():.4f} | 用时: {elapsed:.1f}s")

# 最终评估
model.eval()
with torch.no_grad():
    inputs, targets = get_batch(val_data, batch_size, context_length, device)
    logits = model(inputs)
    val_loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
    print(f"\n训练完成! 最终验证损失: {val_loss.item():.4f}")
    print(f"总训练时间: {time.time() - start_time:.1f}秒")
    
    # 演示保存检查点（可选）
    save_checkpoint(model, optimizer, max_iters, "mini_model_checkpoint.pt")
    print("已保存微型模型检查点: mini_model_checkpoint.pt")