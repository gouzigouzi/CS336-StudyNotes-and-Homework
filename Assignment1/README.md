# Assignment 1 (basics): Building a Transformer LM

## [1 课程作业整体介绍](section1)
接下来我们将开始 Assignment1 的学习

**你将实现的内容：**

1. **字节对编码（BPE）分词器**
   - 从头实现 BPE 算法，用于构建子词（subword）级别的分词器。
   - 在 TinyStories 数据集上训练该分词器。
   - 使用训练好的分词器将文本数据转换为整数 ID 序列，供模型训练使用。

2. **Transformer 语言模型**
   - 完全从头构建一个标准的 Transformer 架构语言模型（如 GPT 风格）。
   - 包括多头自注意力机制（Multi-Head Self-Attention）、前馈网络（Feed-Forward Network）、层归一化（Layer Normalization）、位置编码（Positional Encoding）等组件。
   - 不允许使用 torch.nn 中的层定义（如 Linear、Embedding、LayerNorm 等），但可以使用 torch.nn.Parameter 和容器类（如 Module、ModuleList、Sequential）。
  
3. **交叉熵损失函数与 AdamW 优化器**
   - 手动实现交叉熵损失函数（Cross-Entropy Loss），用于语言模型训练。
   - 从头实现 AdamW 优化器（包括动量、自适应学习率和权重衰减机制），不能使用 torch.optim 中的现成优化器（但可继承 torch.optim.Optimizer 基类）。

4. **支持序列化/加载的训练循环**
   - 实现完整的训练循环，支持：
     - 模型状态保存与加载（state_dict 的保存与恢复）
     - 优化器状态的保存与加载
     - 断点续训功能
   - 训练过程中记录损失、困惑度（perplexity）等指标。

**你将运行的任务：**

1. **在 TinyStories 数据集上训练 BPE 分词器**
   - 使用提供的 TinyStories 文本数据训练你实现的 BPE 分词器。

2. **将数据集转换为整数 ID 序列**
   - 使用训练好的 BPE 分词器对 TinyStories 数据进行编码，生成模型可处理的整数 token ID 序列。

3. **在 TinyStories 上训练 Transformer 语言模型**
   - 使用编码后的数据训练你实现的 Transformer 模型。
   - 训练期间监控训练损失，并在验证集上计算困惑度。

4. **生成样本与评估困惑度**
   - 使用训练好的模型进行文本生成（如通过自回归采样）。
   - 在验证集上计算并报告语言模型的困惑度（Perplexity）。

5. **在 OpenWebText 数据集上训练模型并提交结果**
   - 将模型应用于更大的 OpenWebText 数据集进行训练。
   - 在测试集上评估最终模型的困惑度，并将结果提交至课程指定的排行榜。

---
## [2 分词 (Tokenization)](section2)

---
## [3 Transformer 语言模型架构](section3)

---
## [4 训练Transformer LM](section4)

---
## [5 训练与生成](section5)

---
## [6 生成文本（Generating text）](section6)

---
## [7 Experiments](section7)