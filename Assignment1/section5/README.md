# Assignment 1 (basics): Building a Transformer LM

## 5 训练与生成
我们来到了 Training loop 和 Generating text 部分，我们将最终整合迄今为止构建的主要组件：分词后的数据、模型和优化器。

### 5.1 数据加载（Data Loader）
分词后的数据（例如你在 tokenizer_experiments 中准备的数据）是一个单一的 token 序列 $x = (x_1, \ldots, x_n)$。尽管原始数据可能由多个独立的文档组成（例如不同的网页或源代码文件），一种常见的做法是将所有这些文档连接成一个单一的 token 序列，并在它们之间添加一个分隔符（例如 $\text{<|endoftext|>}$ token）。

数据加载器会将这个长序列转换为一批批的数据流，其中每个批次包含 $B$ 个长度为 $m$ 的序列，以及对应的下一个 token 序列（长度也为 $m$）。例如，当 $B$ = 1，$m$ = 3 时，$([x_2, x_3, x_4], [x_3, x_4, x_5])$ 就是一个可能的批次。

以这种方式加载数据在多个方面简化了训练过程。首先，任何满足 $1 \le i < n − m$ 的位置 $i$ 都可以生成一个有效的训练序列，因此序列的采样非常简单。由于所有训练序列长度相同，无需进行填充操作，这提高了硬件利用率（同时也可以增大批次大小 $B$）。最后，我们也不必一次性将整个数据集全部加载到内存中即可进行训练样本的抽取，从而更容易处理那些可能无法完全放入内存的大型数据集。

---
**问题（data_loading）：实现数据加载（2分）**
交付要求：编写一个函数，该函数接收一个 numpy 数组 $x$（包含 token ID 的整数数组）、batch_size、context_length 和一个 PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'），并返回一对张量：采样的输入序列及其对应的下一个 token 目标。这两个张量的形状都应为 (batch_size, context_length)，包含 token ID，并且都应放置在指定的设备上。

低资源/降级训练提示：在 CPU 上进行数据加载
如果你计划在 CPU 上训练你的语言模型，你需要将数据移动到正确的设备上（同样地，之后你的模型也应使用相同的设备）。如果使用 CPU，可使用设备字符串 'cpu'。

代码可见 [dataloader.py](dataloader.py)

如果数据集太大而无法全部加载到内存中怎么办？我们可以使用一个名为 mmap 的 Unix 系统调用，它能将磁盘上的文件映射到虚拟内存，并在访问对应内存位置时才加载文件内容。因此，你可以“假装”整个数据集已经在内存中了。NumPy 通过 np.memmap 提供了这一功能（或者在使用 np.load 时设置参数 mmap_mode='r'，前提是你之前是用 np.save 保存的数组），这会返回一个类数组对象，只有在你访问具体元素时才会按需加载数据。

在训练过程中从数据集（即 NumPy 数组）采样时，务必以内存映射模式加载数据集（通过 np.memmap 或 np.load 的 mmap_mode='r' 参数，具体取决于你保存数组的方式）。同时，请确保指定的 dtype 与你所加载的数组的原始数据类型一致。为确保安全，建议显式验证内存映射的数据是否正确（例如，检查数据中是否包含超出预期词表大小的非法 token ID 值）

### 5.2 检查点（Checkpointing）
除了加载数据外，我们还需要在训练过程中保存模型。在运行训练任务时，我们常常希望能够在训练中途意外停止后（例如由于任务超时、机器故障等）恢复训练。即使一切顺利，我们也可能希望之后能够访问训练过程中的中间模型（例如，事后研究训练动态、从不同训练阶段的模型中生成样本等）。

一个检查点（checkpoint）应包含所有能够恢复训练所需的状态。最基本的是，我们必须能够恢复模型的权重。如果使用了带有状态的优化器（例如 AdamW），我们还需要保存优化器的状态（例如，AdamW 中的动量估计值）。最后，为了能够恢复学习率调度，我们还需要知道训练停止时所处的迭代次数。PyTorch 使得保存这些信息变得非常简单：每个 `nn.Module` 都有一个 `state_dict()` 方法，它会返回一个包含所有可学习参数的字典；之后我们可以通过对应的 `load_state_dict()` 方法恢复这些权重。优化器 `torch.optim.Optimizer` 也同样支持 `state_dict()` 和 `load_state_dict()` 方法。最后，`torch.save(obj, dest)` 可以将一个对象（例如，字典，其值中包含张量，也可以包含整数等普通 Python 对象）序列化并保存到文件（路径）或类文件对象中，之后可以通过 `torch.load(src)` 将其重新加载回内存。

---
**问题（checkpointing）：实现模型检查点保存与加载（1分）**
实现以下两个函数，用于保存和加载模型检查点：
`def save_checkpoint(model, optimizer, iteration, out)` 应将前三个参数中的所有状态保存到文件类对象 out 中。你可以使用模型和优化器的 `state_dict` 方法获取它们的相关状态，并使用 `torch.save(obj, out)` 将 obj 保存到 out（PyTorch 支持路径或文件类对象）。一个常见的做法是让 obj 是一个字典，但你也可以使用任何格式，只要之后能够成功加载你的检查点即可。

该函数需要以下参数：
- model: torch.nn.Module
- optimizer: torch.optim.Optimizer
- iteration: int
- out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]

`def load_checkpoint(src, model, optimizer)` 应从 src（路径或文件类对象）加载检查点，并恢复模型和优化器的状态。你的函数应返回保存在检查点中的迭代次数。你可以使用 `torch.load(src)` 来读取你在 save_checkpoint 中保存的内容，并通过模型和优化器的 `load_state_dict` 方法将它们恢复到之前的状态。

该函数需要以下参数：
- src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
- model: torch.nn.Module
- optimizer: torch.optim.Optimizer

代码可见 [checkpoint.py](checkpoint.py)

### 5.3 训练循环
现在，是时候将你之前实现的所有组件整合到你的主训练脚本中了。为了方便后续多次运行训练（以研究不同参数选择对训练的影响），建议将训练过程配置为可通过命令行参数轻松启动，并支持不同的超参数设置。

---
**问题（training_together）：整合所有组件（4分）**
交付要求： 编写一个脚本，运行训练循环，使用用户提供的输入来训练你的模型。具体来说，我们建议你的训练脚本至少具备以下功能：
- 可配置和控制各种模型及优化器的超参数。
- 使用 np.memmap 以内存高效的方式加载大型训练和验证数据集。
- 将检查点序列化保存到用户指定的路径。
- 定期记录训练和验证性能（例如，输出到控制台和/或外部服务如 Weights and Biases）。

代码可见 [training_loop.py](training_loop.py)
