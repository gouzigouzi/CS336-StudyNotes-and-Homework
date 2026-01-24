# Assignment 1 (basics): Building a Transformer LM

## 6 生成文本（Generating text）
现在我们已经可以训练模型了，最后需要实现的功能就是如何从模型中生成文本。回想一下，语言模型接收一个（可能是批量的）长度为 sequence_length 的整数序列，并输出一个大小为 (sequence_length $\times$ 词汇表大小) 的矩阵，其中序列的每个位置对应一个概率分布，用于预测该位置之后的下一个词。接下来，我们将编写几个函数，将这种输出转化为生成新序列的采样方法。

**Softmax** 
按照常规，语言模型的输出是最终线性层的输出（即“logits”），因此我们需要通过 softmax 操作将其转换为归一化的概率分布。

**解码** 
为了从模型生成文本（即解码），我们会先提供一段前缀 token 序列（即“提示”或 prompt），让模型输出一个在整个词汇表上的概率分布，以预测序列的下一个词。然后，我们将从该词汇表的概率分布中进行采样，以确定下一个输出 token。

具体来说，解码过程的一步应输入一个序列 $x_{1\ldots t}$，并通过以下公式返回一个标记 $x_{t+1}$：
$$
P(x_{t+1}=i \mid x_{1:t})
= \frac{\exp(v_i)}{\sum_j \exp(v_j)} .
$$

其中
$$
v = \mathrm{TransformerLM}(x_{1:t})_t \in \mathbb{R}^{\text{vocab\_size}} .
$$

这里，$\mathrm{TransformerLM}$ 是我们的模型，它以长度为 sequence_length 的序列作为输入，并输出一个大小为 (sequence_length × vocab_size) 的矩阵。我们取该矩阵的最后一个元素，因为我们关注的是在第 $t$ 个位置的下一个词预测。

这为我们提供了一个基本的解码器，通过不断从这些单步条件概率中进行采样（将先前生成的输出标记添加到下一个解码时间步的输入中），直到生成序列结束标记（或达到用户指定的最大生成标记数量）为止。

**解码器的实现** 
我们将使用小型模型进行实验，而小型模型有时会生成质量很低的文本。两种简单的解码技巧可以帮助解决这些问题。首先，在温度调节（temperature scaling）中，我们通过引入一个温度参数 $\tau$ 来调整我们的 softmax 函数，新的 softmax 公式为：
$$
\mathrm{softmax}(v,\tau)_i
= \frac{\exp(v_i/\tau)}{\sum_{j=1}^{\text{vocab\_size}} \exp(v_j/\tau)} .
$$

请注意，当设置 $\tau\rightarrow 0$ 时，会使向量 $v$ 中的最大元素占据主导地位，softmax 的输出会变成一个集中在该最大元素上的 one-hot 向量。

其次，另一种技巧是 nucleus（核采样）或 top-p 采样，它通过截断低概率词汇来修改采样分布。设 $q$ 是从一个（经过温度缩放的）softmax 得到的概率分布，其大小为 vocab_size。使用超参数 $p$ 的 nucleus 采样根据以下公式生成下一个 token：
$$
P(x_{t+1}=i \mid q) =
\begin{cases}
\dfrac{q_i}{\sum_{j\in V(p)} q_j}, & i\in V(p),\\[6pt]
0, & \text{otherwise}.
\end{cases}
$$

其中，$V(p)$ 是满足 $\sum_{j\in V(p)} q_j \ge p$ 的最小索引集合。通过首先按概率大小对分布 $q$ 进行排序，然后选择最大的词汇项累加概率，直到累积概率达到目标值 $p$ 来计算这个集合。这样能避免采到太低概率的怪词，同时又保留随机性。

---
**问题（decoding）：解码（3分）**
交付内容：实现一个从你的语言模型中进行解码的函数。我们建议你支持以下功能：
- 为用户提供提示词生成补全内容（即输入一段文本 $x_{1\ldots t}$，然后采样生成后续内容，直到生成结束标记 ）。
- 允许用户控制生成的最大 token 数量。
- 给定一个指定的温度值，对预测的下一个词分布应用 softmax 温度缩放后再进行采样。
- 支持 Top-p 采样（Holtzman 等，2020；也称为核采样），给定用户指定的阈值。

代码可见 [inference.py](inference.py)
