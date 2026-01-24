from hw3.softmax import softmax
import torch


class CrossEntropyLoss:
    def __init__(self,inputs,targets):
        """
        初始化交叉熵损失计算器
        """
        self.inputs = inputs        # 模型输出的原始 logits (batch_size, vocab_size)
        self.targets = targets      # 真实标签索引 (long tensor) (batch_size,)
        self.vocab_size = inputs.shape[1]
        self.batch_size = inputs.shape[0]

    def forward(self):
        """
        前向计算交叉熵损失
        步骤：softmax -> 取真实类概率 -> 负对数求和
        """
        y_pred = softmax(self.inputs,1)  # 对每行做 softmax 得预测概率
        # 提取真实标签对应的概率 p = y_pred[i, targets[i]]
        p = y_pred[range(self.batch_size),self.targets]
        return -torch.sum(torch.log(p))


