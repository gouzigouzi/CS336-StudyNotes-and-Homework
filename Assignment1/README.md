# Assignment 1 (basics): Building a Transformer LM

## 1 课程作业整体介绍

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

## 2 分词 (Tokenization)
### 2.1 Unicode标准
在本作业的第一部分，我们将实现并训练一个**字节级字节对编码（Byte-Pair Encoding, BPE）分词器**，该方法基于 Sennrich 等人（2016）和 Wang 等人（2019）的工作。与传统的基于字符或子词的 BPE 不同，我们将以**字节（byte）为基本单位**进行分词器的构建。

具体来说，我们会将任意 Unicode 字符串首先转换为对应的字节序列（使用 UTF-8 编码），然后在这个字节序列上运行 BPE 算法。这样做的好处是：
- 可以自然地处理所有 Unicode 字符（包括英文、中文、表情符号等），无需显式地维护庞大的词汇表；
- 模型具备开箱即用的多语言支持能力；
- 避免了未知词（OOV, out-of-vocabulary）问题，因为任何字符串都可以被编码为字节序列。

最终，该分词器能够将输入文本转换为整数 token ID 序列，供后续的 Transformer 语言模型训练使用。

**Unicode 与字节编码简介**
在 Python 中，字符串以 Unicode 形式存储。每个字符对应一个唯一的码点（code point），可以通过 ord() 函数获取其整数值，例如：
```python
ord('s')   # 输出: 115    
ord('牛')  # 输出: 29275
```
反之，使用 chr() 可以将码点转换回字符：
```python
chr(115)    # 输出: 's'    
chr(29275)  # 输出: '牛'
```
为了转换为字节序列，我们可以使用 UTF-8 编码：
```python
"Hello 😂".encode('utf-8')  # 输出: b'Hello \xf0\x9f\x98\x82'
```
这会返回一个字节序列（bytes 类型），其中每个元素是 0 到 255 之间的整数。BPE 分词器将在这样的字节序列上进行训练。

**问题（1分）**
(a) chr(0)返回什么Unicode字符？
```python
chr(0)  # 输出: '\x00'
```
(b) 该字符的字符串表示与打印表示有何不同？
(c) 当该字符出现在文本中会发生什么？
```python
l = "this is a test" + chr(0) + "string"
l  # 'this is a test\x00string'
print(l)  # 输出: this is a teststring
```

### 2.2 Unicode编码
尽管 Unicode 标准为每个字符定义了唯一的代码点（即一个整数），但直接在 Unicode 代码点上训练分词器并不可行。主要原因有两个：一是 Unicode 字符总数庞大（目前已定义的字符超过 15 万个），导致词汇表规模过大；二是大多数字符在实际文本中极为罕见，造成词汇表高度稀疏，不利于模型学习和泛化。

为解决这一问题，我们转而使用 Unicode 编码方案 将文本转换为字节序列，并在字节级别上构建分词器。Unicode 定义了多种编码格式，其中最常用的是 UTF-8、UTF-16 和 UTF-32。在这些编码中，UTF-8 是当前互联网上最主流的编码方式，据估计超过 98% 的网页都采用 UTF-8。

UTF-8 的一个重要特性是：它将每个 Unicode 字符编码为 1 到 4 个字节的序列（对于基本 ASCII 字符仅用 1 字节，而中文、表情符号等则使用 3 或 4 字节），兼容 ASCII 且可变长，高效且广泛支持。
```python
test_string = "hello! 天海!"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)  # 输出: b'hello! \xe5\xa4\xa9\xe6\xb5\xb7!'
print(type(utf8_encoded))  # 输出：<class 'bytes'>

# 拆解字节值（0-255整数）
list(utf8_encoded)  

# 验证可逆性
print(len(test_string), len(utf8_encoded))  # 输出: 10  14
print(utf8_encoded.decode("utf-8"))  # 输出: 'hello! 天海!'
```

**问题 (3 分)**
(a) 为什么我们更倾向于在UTF-8编码的字节上训练分词器，而不是UTF-16或UTF-32？比较这些编码对不同输入字符串的输出可能有所帮助。

我们更倾向用 UTF-8，因为它**字节为基本单位、与 ASCII 兼容且对英文/常见文本更省空间**：ASCII 字符在 UTF-8 中只占 1 字节，而 UTF-16 至少 2 字节、UTF-32 固定 4 字节，会让序列更长、训练和存储更低效。并且 UTF-16/32 的字节序与代理项等细节更复杂，用“字节级”BPE 时更容易引入实现与跨平台一致性问题。


(b) 考虑以下（错误的）函数，其目的是将UTF-8字节串解码为Unicode字符串。为什么这个函数是错误的？提供一个会产生错误结果的输入字节串示例。
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```
这个函数错在它把 UTF-8 的每个“字节”当成一个完整字符去单独解码。但 UTF-8 是变长编码，很多非 ASCII 字符会由 2–4 个字节共同组成；把它们拆开逐字节`decode("utf-8")`时，每个单独字节往往不是合法的 UTF-8 序列，会报错或得到错误结果。
```python
decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
# decode_utf8_bytes_to_str_wrong("café".encode("utf-8")) 错误
```

### 2.3 子字标记化（subword tokenization）
虽然字节级标记化能够有效缓解单词级标记器面临的词汇表外问题，但将文本分解为单个字节会导致输入序列过长。例如，一个包含10个单词的句子在单词级模型中可能仅对应10个标记，而在字节级模型中却可能膨胀至50个甚至更多标记，具体取决于单词长度。这种扩展显著增加了模型每一步的计算量，拖慢训练速度。同时，过长的序列也给语言建模带来挑战，因为它在数据中引入了更复杂的长期依赖关系。

子字标记化（subword tokenization）则介于单词级和字节级之间，提供了一种折中方案。与仅有256个条目的字节级词汇表不同，子词标记器通过扩大词汇量来更高效地压缩原始字节序列。其核心思想是：如果某些字节序列（如 b'the'）在训练数据中频繁出现，就将其合并为一个单独的标记，从而将原本多个字节组成的序列压缩为一个单元。这样既能保持对罕见词和未知词的处理能力，又能显著缩短平均序列长度。

如何选择这些子词单元？Sennrich 等人（2016）提出采用字节对编码（Byte Pair Encoding, BPE），这是一种源自数据压缩技术的算法。BPE 通过迭代地查找并合并出现频率最高的相邻字节对，逐步构建出一组高效的子词单元。每次合并都会引入一个新的符号来代表该字节对，并将其加入词汇表。这一过程持续进行，直到达到预设的词汇表大小。由于BPE优先合并高频模式，因此最终的词汇表能最大程度地提升整体压缩效率——常见词或词片段更可能被表示为单一标记。

在本任务中，我们将实现一种基于字节的BPE分词器，其词汇项由原始字节及其合并后的序列表示。这种方法结合了字节级分词器的鲁棒性与子词级的高效性，在处理未登录词的同时保持合理的序列长度。整个构建词汇表的过程也被称为“训练”BPE分词器，是实现高效文本表示的关键步骤。

### 2.4 BPE分词器训练
BPE分词器的训练包含三个步骤：初始化、预分词和合并。

#### 2.4.1 词表初始化
首先进行词汇表的初始化，BPE分词器的词汇表本质上是一个从字节字符串到整数ID的一一映射。由于我们训练的是字节级BPE分词器，初始词汇表包含所有256个可能的字节值，因此初始大小为256。

#### 2.4.2 预分词
接下来是预分词阶段。理论上，我们可以直接在原始字节序列上统计相邻字节对的出现频率并开始合并，但这样每次合并都需要遍历整个语料库，计算成本极高。此外，若不加处理地跨文本合并字节，可能导致仅因标点不同而语义相近的词被拆分为完全不同的一组标记，例如“dog!”和“dog.”会被视为完全无关的标记，不利于模型学习其语义一致性。为缓解这一问题，我们先对语料进行预分词。预分词可以看作是对文本的一次粗粒度切分，有助于高效统计字节对的共现频率。例如，如果单词“text”作为预分词出现了10次，那么其中的字节对‘t’和‘e’的共现次数就可以一次性增加10次，而无需重复扫描整个语料。每个预分词以UTF-8编码的字节序列表示。

原始BPE实现中采用简单的空格分割`s.split(" ")`，而我们则采用GPT-2所使用的正则表达式预分词器（来自OpenAI的tiktoken项目），其模式为：`r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"`。例如对句子“some text that i'll pre-tokenize”进行切分，结果为：['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']。在实际编码中，建议使用`re.finditer`而非`re.findall`，以便在构建预分词频次映射时避免额外存储中间列表。

#### 2.4.3 BPE合并计算
完成预分词后，进入BPE合并的计算阶段。此时，每个预分词已被表示为字节序列，算法开始迭代地统计所有相邻字节对的出现频率，并选择频率最高的字节对进行合并。例如，若字节对（A, B）是当前最频繁的组合，则将所有连续出现的A和B替换为一个新的合并标记“AB”，并将该新标记加入词汇表。由于初始词汇表大小为256，最终词汇表的大小等于256加上训练过程中执行的合并操作数量。为了提升效率，在统计字节对时不会跨越预分词边界进行合并。当多个字节对频率相同时，采用字典序较大的那一对作为胜出者。例如，若（"A", "B"）、（"A", "C"）、（"B", "ZZ"）和（"BA", "A"）频率相同，则选择**字典序最大**的（"BA", "A"）进行合并。

此外，某些特殊字符串用于表示元信息，例如文档边界或序列结束。这类字符串应作为“特殊标记”被整体保留，绝不允许被拆分。因此，这些特殊标记必须提前加入词汇表，并分配固定的ID。例如表示序列结束的字符串，必须始终对应单一标记，以便模型在生成文本时能准确判断何时停止。Sennrich等人（2016）的原始论文中给出了BPE训练的非优化版本，适合初学者实现以理解基本流程。

以论文中的示例说明：假设语料包含以下文本：
```
low low low low low    
lower lower widest widest widest    
newest newest newest newest newest newest    
```
并设定<|endoftext|>为特殊标记。初始词汇表包括<|endoftext|>和全部256个字节。

预分词采用空格分割，得到频次统计：low（5次）、lower（2次）、widest（3次）、newest（6次），可表示为字节元组的频数字典，如{(b'l', b'o', b'w'): 5, ...}。随后统计所有相邻字节对的频率：lo（7次）、ow（7次）、we（8次）、er（2次）、wi（3次）、id（3次）、de（3次）、es（9次）、st（9次）、ne（6次）、ew（6次）。其中（es）和（st）频率最高且相同，按字典序选择更大的（st）进行合并。于是，所有包含“st”的词如“widest”和“newest”中的“s”和“t”被合并为新标记“st”。第二轮中，“e”与“st”组合出现9次，成为最高频对，合并为“est”。继续此过程，后续合并依次为“ow”、“low”、“west”、“ne”等。若仅执行6次合并，最终新增标记为['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']。更新计算后的词汇表如下所示。
```
[<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]
```
此时，“newest”将被切分为[ne, west]两个标记。这一机制在保持对未知词处理能力的同时，有效压缩了序列长度，提升了模型效率。

### 2.5 BPE分词器训练实验
现在我们在 TinyStories 数据集上训练字节级 BPE 分词器。
https://huggingface.co/datasets/roneneldan/TinyStories/tree/main

**并行化预分词**
预分词是主要性能瓶颈，可使用 multiprocessing 库进行并行加速。建议在确保分块边界位于特殊 token 起始位置的前提下对语料分块。可参考以下链接中的示例代码获取分块边界，用于跨进程任务分配：https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py%E3%80%82

该分块方式始终有效，因为我们不会在文档边界执行合并操作。本作业中可始终采用此方法，无需考虑极端情况（如超大语料中不含标记的情形）。

**预分词前移除特殊token**
在使用正则表达式`re.finditer`进行预分词前，应先从语料（或分块）中移除所有特殊 token。必须确保在特殊 token 处分割文本，防止跨文档合并。例如，对于 [文档1] [文档2] 的语料，应在 处分割，分别对两部分进行预分词。可通过`re.split`实现，使用`"|".join(special_tokens)`作为分隔符（注意使用`re.escape`处理特殊字符，避免`|`等符号引发问题）。

**优化合并步骤**
朴素的 BPE 训练实现效率较低，因每次合并后需重新遍历所有字节对以找最高频对。实际上，仅与被合并字节对相邻的计数会发生变化。因此，可通过维护字节对计数的索引结构，并在每次合并后增量更新相关计数，显著提升速度。尽管该缓存机制能大幅加速，但 BPE 合并步骤在 Python 中无法并行化。

**低资源/降级提示：性能分析**
建议使用 cProfile 或 scalene 等工具进行性能分析，定位瓶颈并集中优化关键部分。

**低资源/降级提示：降级训练**
不要直接在完整 TinyStories 数据集上训练。建议先在小规模子集（“调试数据集”）上实验，例如使用包含 2.2 万篇文档的验证集（而非完整的 212 万篇）。这是一种通用开发策略：通过使用更小数据、更小模型来加速迭代。选择调试集规模和超参数时需权衡：应足够大以复现真实瓶颈，又不宜过大导致耗时过长。

**问题（15分）BPE 分词器训练**
交付要求：编写一个函数，根据输入的文本文件路径训练（字节级）BPE分词器。你的BPE训练函数应至少处理以下输入参数：
- input_path: str BPE分词器训练数据文本文件的路径。
- vocab_size: int 定义最终词汇表最大大小的正整数（包括初始字节词汇表、合并产生的词汇表项和任何特殊token）。
- special_tokens: list[str] 需要添加到词汇表中的字符串列表。这些特殊token不会影响BPE训练过程。

你的BPE训练函数应返回最终的词汇表和合并记录：
- vocab: dict[int, bytes] 分词器词汇表，一个从整数（词汇表中的token ID）到字节（token字节）的映射。
- merges: list[tuple[bytes, bytes]] 训练产生的BPE合并记录列表。每个列表项是一个字节元组(, )，表示与被合并。
  
代码可见[hw1_1bpe_tokenzier_train.py](hw1/hw1_1bpe_tokenzier_train.py)

**问题（2 分）TinyStories 的 BPE 训练**
(1) 在 TinyStories 数据集上训练一个字节级 BPE 分词器，最大词汇表大小为 10,000。确保将 TinyStories 的 </s> 特殊标记加入词汇表。将生成的词汇表和合并规则序列化保存到磁盘以便进一步检查。训练耗时多少小时，占用多少内存？词汇表中最长的 token 是什么？这合理吗？
资源要求：≤ 30 分钟（不使用 GPU），≤ 30GB 内存
提示：通过在预分词阶段使用多进程，并结合以下两个事实，你应该能在 2 分钟内完成 BPE 训练：
(a) 数据文件中使用 </s> 标记来分隔文档。
(b) </s> 标记在应用 BPE 合并之前已被作为特殊情况处理。
交付内容：一到两句话的回复。
(2) 对你的代码进行性能分析。分词器训练过程中哪一部分耗时最长？
交付内容：一到两句话的回复。

### 2.6 BPE分词器训练实验
在作业的前一部分中，我们实现了一个函数，用于在输入文本上训练 BPE 分词器，以获得分词器词汇表和 BPE 合并列表。现在，我们将实现一个 BPE 分词器，它加载提供的词汇表和合并列表，并使用它们对文本进行编码和解码到标记 ID 或从标记 ID 中进行编码和解码。

#### 2.6.1 对文本进行编码
BPE 对文本进行编码的过程反映了我们训练 BPE 词汇的方式。整个过程可分为以下几个主要步骤：

**第1步：预分词化（Pre-tokenization）**
我们首先使用预分词器（pre-tokenizer）将输入文本切分为“预分词”（pre-tokens）。常见的策略是按空格或标点切分，但保留边界信息（例如，将空格作为下一个词的前缀）。每个预分词将被独立处理。

然后，我们将每个预分词转换为其对应的 UTF-8 字节序列（bytes），作为后续合并操作的基本单位。
```
🔍 示例：字符串 'the cat ate' 被切分为 ['the', ' cat', ' ate']，每个元素都以字节形式表示。
```

**第2步：应用 BPE 合并规则**
对于每个预分词，我们将其初始字节序列按照训练阶段学到的合并规则列表（merges）逐步合并。合并顺序至关重要——必须严格按照训练时产生的顺序依次尝试。

每次查找当前序列中是否存在可应用的合并对（相邻且完全匹配）
若存在，则执行合并，生成更长的子词单元
重复此过程，直到无法再应用任何规则
⚠️ 注意：合并不会跨越预分词边界。也就是说，不同预分词之间的字节不会被合并，保证了分词的局部性与可预测性。

**第3步：映射为 Token ID**
当每个预分词完成所有可能的合并后，得到一组最终的子词单元（subword units）。我们通过查表的方式，将这些子词单元映射为词汇表中的整数 ID，形成最终的 token ID 序列。

#### 2.6.2 详细案例解析：'the cat ate' 的完整编码过程
为了更深入理解上述流程，下面我们对输入字符串 'the cat ate' 进行端到端的 BPE 编码演示。

**输入信息**
- 输入字符串：'the cat ate'
- 词汇表（Vocabulary）：
  ```   
  0: b' ',     # 空格    
  1: b'a',    
  2: b'c',    
  3: b'e',    
  4: b'h',    
  5: b't',    
  6: b'th',    
  7: b' c',   # 空格 + c    
  8: b' a',   # 空格 + a    
  9: b'the',    
  10: b' at'   # 空格 + a + t    
  ```
- 合并规则（Merges）（按优先级顺序）：
  ```  
  (b't', b'h'),      # → b'th'    
  (b' ', b'c'),      # → b' c'    
  (b' ', b'a'),      # → b' a'    
  (b'th', b'e'),     # → b'the'    
  (b' a', b't')      # → b' at'    
  ```
- 预分词策略：按空格分割，空格归属于后续词（即作为前缀）

**步骤一：预分词（Pre-tokenization）**
原始字符串：'the cat ate'
切分结果：
```
['the', ' cat', ' ate']
```
解释：
- 'the'：无前导空格
- ' cat'：包含前导空格
- ' ate'：包含前导空格

💡 此策略影响后续合并行为，因为空格被视为字符的一部分。

**步骤二：逐个预分词应用 BPE 合并**
✅ 预分词 1: 'the'
1. 初始分解：[b't', b'h', b'e']
2. 应用合并：
   - (b't', b'h') → 合并为 [b'th', b'e']
   - (b'th', b'e') → 合并为 [b'the']
3. 查表得 ID：b'the' → ID 9
4. 输出：[9]
✅ 完成。

✅ 预分词 2: ' cat'
1. 初始分解：[b' ', b'c', b'a', b't']
2. 应用合并：
   - (b' ', b'c') → 合并为 [b' c', b'a', b't']
   - 其他规则无法应用：
     - (b' ', b'a')：当前没有独立的 b' ' 和 b'a' 相邻
     - (b' a', b't')：需要 b' a'，但此处是 b'a' 且前接 b' c'
3. 查表得 ID：
   - b' c' → 7
   - b'a' → 1
   - b't' → 5
4. 输出：[7, 1, 5]
✅ 完成。

✅ 预分词 3: ' ate'
1. 初始分解：[b' ', b'a', b't', b'e']
2. 应用合并：
   - (b' ', b'a') → 合并为 [b' a', b't', b'e']
   - (b' a', b't') → 合并为 [b' at', b'e']
3. 查表得 ID：
   - b' at' → 10
   - b'e' → 3
4. 输出：[10, 3]
✅ 完成。

**步骤三：拼接所有 token ID**
将各预分词的编码结果串联：
- 'the' → [9]
- ' cat' → [7, 1, 5]
- ' ate' → [10, 3]
最终编码结果：
```
[9, 7, 1, 5, 10, 3]
```

#### 2.6.3 关键要点总结
| 要点 | 说明 |
|  ---- | ---- |
| 🔁 合并顺序决定结果 | 必须严格按照训练时生成的顺序尝试合并，顺序不同可能导致不同输出 |
| 🚫 不跨预分词边界合并 | 即使两个预分词末尾和开头可以合并（如 'the' 和 ' cat' 中的 'e' 和 ' '），也不会发生跨词合并 |
| ⚠️ 上下文敏感性 | 相同字符组合因上下文不同可能被不同编码（如 'at' 在 'cat' 和 ' at' 中表现不同） |
| 🧩 空格处理方式至关重要 | 是否保留、作为前缀/后缀，直接影响合并路径 |
| 📇 词汇表完整性 | 所有合并后的子词必须存在于词汇表中，否则无法映射为 ID |

#### 2.6.4 解码：从 Token ID 到文本
编码的逆过程称为解码（Decoding），即将 token ID 序列还原为原始文本。

解码步骤：
1. 查表反向映射：将每个 ID 转换回对应的字节串（如 9 → b'the'）
2. 拼接字节串：按顺序连接所有字节
3. 解码为字符串：将拼接后的字节序列用 UTF-8 解码为 Unicode 字符串
4. 处理空格规范化（可选）：根据需要去除多余空格或调整格式

示例：解码 [9, 7, 1, 5, 10, 3]
1. 查表：
   - 9 → b'the'
   - 7 → b' c'
   - 1 → b'a'
   - 5 → b't'
   - 10 → b' at'
   - 3 → b'e'
2. 拼接字节：
```
b'the' + b' c' + b'a' + b't' + b' at' + b'e'
```

**问题（15分）实现分词器**
可交付成果：实现一个 Tokenizer 类，该类在给定词汇表和合并列表的情况下，将文本编码为整数 ID，并将整数 ID 解码为文本。分词器还应支持用户提供的特殊令牌（如果词汇表尚不存在，则将其附加到词汇表中）。我们推荐以下界面：

`def __init__(self, vocab, merges, special_tokens=None)`
根据给定的词汇表、合并规则列表以及（可选的）特殊标记列表构建一个分词器。该函数应接受以下参数：
- vocab: 一个从整数 ID 映射到字节串的字典（dict[int, bytes]）
- merges: 一个包含字节对元组的列表，表示 BPE 合并规则（list[tuple[bytes, bytes]]）
- special_tokens: 一个字符串列表，表示特殊标记，可选（list[str] | None = None）

`def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)`
一个类方法，用于从序列化的词汇表文件和合并规则文件（格式与你训练 BPE 代码输出的格式相同）构建并返回一个 Tokenizer，同时可选地接收一个特殊标记列表。该方法应接受以下额外参数：
- vocab_filepath: 词汇表文件的路径（str）
- merges_filepath: 合并规则文件的路径（str）
- special_tokens: 一个字符串列表，表示特殊标记，可选（list[str] | None = None）

`def encode(self, text: str) -> list[int]`
将输入文本编码为一个 token ID 序列。

`def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`
给定一个字符串的可迭代对象（例如，一个 Python 文件句柄），返回一个生成器，惰性地逐个生成 token ID。这对于内存受限情况下对大型文件进行高效分词是必需的。

`def decode(self, ids: list[int]) -> str`
将一个 token ID 序列解码为原始文本。要使用我们提供的测试用例验证你的 Tokenizer 实现，你需要先实现 [adapters.get_tokenizer] 中的测试适配器，然后运行 uv run pytest tests/test_tokenizer.py。你的实现应能通过所有测试。

代码可见[hw1_2_implementing_the_tokenizer.py](hw2/hw1_2_implementing_the_tokenizer.py)



## 3 Transformer 语言模型架构
一个语言模型接收一批整数形式的标记ID序列作为输入（即，形状为 (batch_size, sequence_length) 的 torch.Tensor），并返回一个（批处理的）在词汇表上的归一化概率分布（即，形状为 (batch_size, sequence_length, vocab_size) 的 PyTorch Tensor），其中预测的分布是针对每个输入标记的下一个词。在训练语言模型时，我们使用这些对下一个词的预测，计算实际下一个词与预测下一个词之间的交叉熵损失。在推理阶段生成文本时，我们使用前馈网络最后一步（即序列中的最后一个时间步）预测的下一个词分布来生成序列中的下一个标记（例如，选择概率最高的标记，或从分布中采样等），将生成的标记添加到输入序列中，然后重复此过程。

![](../figures/fig1.png)
在本作业的这一部分，你将从零开始构建这个Transformer语言模型。我们首先会给出模型的高级概述，然后再逐步详细介绍各个组成部分。

### 3.1 Transformer 语言模型
给定一个标记ID序列，Transformer语言模型首先使用输入嵌入（input embedding）将标记ID转换为稠密向量，然后将这些嵌入后的标记依次通过 num_layers 个 Transformer 块进行处理，最后应用一个可学习的线性投影（称为“输出嵌入”或“语言模型头”，即 LM head），生成预测的下一个标记的 logits。
![](../figures/fig2.png)

#### 3.1.1 词元嵌入
在第一步中，Transformer 将（批处理的）标记ID序列嵌入为一系列向量，这些向量包含标记本身的信息（如图中的红色块所示）。
更具体地说，给定一个标记ID序列，Transformer 语言模型使用一个词元嵌入层（token embedding layer）生成一个向量序列。每个嵌入层接收一个形状为 (batch_size, sequence_length) 的整数张量，并输出一个形状为 (batch_size, sequence_length, d_model) 的向量序列。

#### 3.1.2 预归一化 Transformer 块
嵌入之后，激活值会经过多个结构相同的神经网络层进行处理。一个标准的 **decoder-only** Transformer 语言模型由 num_layers 个相同的层（通常称为 Transformer “块”）组成。
每个 Transformer 块接收一个形状为 (batch_size, sequence_length, d_model) 的输入，并输出一个相同形状的张量 (batch_size, sequence_length, d_model)。每个块通过自注意力机制在序列中聚合信息，并通过前馈网络层对其进行非线性变换。

### 3.2 输出归一化与嵌入
经过 num_layers 个 Transformer 块处理后，我们将获取最终的激活值，并将其转换为词汇表上的概率分布。

我们将实现“预归一化”（pre-norm）Transformer块（详见第3.5节），这种结构额外要求在最后一个Transformer块之后使用层归一化（Layer Normalization，详见下文），以确保输出具有适当的尺度。

在此归一化之后，我们将使用一个标准的可学习线性变换，将 Transformer 块的输出转换为预测的下一个标记的 logits，参见例如 Radford 等 [2018] 的公式2，如下所示:
$$
\mathrm{FFN}(x) = \max(0, xW_1 + b_1)\,W_2 + b_2.
$$

### 3.3 备注：批处理、Einsum 与高效计算
在整个 Transformer 模型中，我们会对许多类似批次的输入执行相同的计算。以下是一些例子：
- 批次中的元素：我们对每个批次中的样本都应用相同的 Transformer 前向操作。
- 序列长度：像 RMSNorm 和前馈网络这样的“逐位置”（position-wise）操作，在序列的每个位置上都以相同方式执行。
- 注意力头：在“多头注意力”机制中，注意力操作会在多个注意力头之间进行批处理。

因此，我们需要一种使用方便且高效的方式来执行这些操作，既能充分利用 GPU 的并行计算能力，又便于阅读和理解。许多 PyTorch 操作可以接受张量前端额外的“类批次”维度，并高效地在这些维度上重复或广播操作。

例如，假设我们要执行一个逐位置的批处理操作。我们有一个数据张量 D，其形状为 (batch_size, sequence_length, d_model)，并希望将其与一个形状为 (d_model, d_model) 的矩阵 A 进行批量向量-矩阵乘法。在这种情况下，D @ A 会自动执行批量矩阵乘法，这是 PyTorch 中一种高效的原语操作，其中 (batch_size, sequence_length) 维度被视为批处理维度。

正因为如此，最好假设你的函数可能会接收到额外的类批次维度，并始终将这些维度保留在 PyTorch 张量形状的最前面。为了组织张量以便能够以这种方式进行批处理，可能需要多次使用 view、reshape 和 transpose 操作来调整形状。这可能会比较繁琐，也常常导致代码难以阅读，且难以追踪张量的实际形状。

更便捷的选择是使用 torch.einsum 中的 einsum 记法，或采用框架无关的库如 einops 或 einx。两个核心操作是：
- **einsum**：可对任意维度的张量进行缩并（tensor contraction）；
- **rearrange**：可重新排列、拼接或分割张量维度。

事实上，机器学习中的大多数操作都可以归结为**维度变换**和**张量缩并**，外加偶尔的（通常是逐元素的）非线性函数。这意味着使用 einsum 记法可使代码更清晰、灵活。

我们强烈建议在课程中学习并使用 einsum 记法：

初学者请使用 einopshttps://einops.rocks/1-einops-basics/%EF%BC%9B
熟悉 einops 的同学可进一步学习更通用的 einx。

以下是一些使用 einsum 记法的示例，作为 einops 文档的补充。

```python
import torch
from einops import rearrange, einsum
```

计算张量 D（形状 [batch, seq, d_in]）和矩阵 A（形状 [d_out, d_in]）的乘积，结果应为 [batch, seq, d_out]。
1. **基础方法**：@ 运算符直接计算，需手动转置 A
2. **显式维度**：einops 第一种写法明确命名所有维度（如 batch seq d_in），可读性最佳
3. **省略号语法**：... d_in 自动匹配前置维度，适合高维张量（如后续的多头注意力案例）

验证部分通过 `torch.allclose` 确认所有方法数值等价。最后扩展展示了多头注意力场景，其中 ... 语法能优雅处理 [batch, heads, seq, dim] 的复杂维度，避免了繁琐的维度命名。einops 的优势在于：**维度意图更直观**，且省略号写法**泛用性更强**，特别适合深度学习中的高维张量操作。

```python
# 设定随机种子保证可复现性
torch.manual_seed(42)

# 定义输入张量
batch_size = 3
seq_len = 5
d_in = 4
d_out = 2

# 创建测试数据
D = torch.randn(batch_size, seq_len, d_in)  # 形状: [batch, sequence, d_in]
A = torch.randn(d_out, d_in)                # 形状: [d_out, d_in]

# 方法1: 使用普通矩阵乘法 @
Y_matmul = D @ A.T                          # A.T形状变为 [d_in, d_out]
print("矩阵乘法结果形状:", Y_matmul.shape)  # 输出: [batch, sequence, d_out]

# 方法2: 使用torch.einsum
Y_torch = torch.einsum("b s i, o i -> b s o", D, A)
print("torch.einsum结果形状:", Y_torch.shape)

# 方法3: 使用einops.einsum (第一种形式)
Y_einops1 = einsum(D, A, "batch seq d_in, d_out d_in -> batch seq d_out")
print("einops形式1结果形状:", Y_einops1.shape)

# 方法4: 使用einops.einsum (第二种形式，使用...)
Y_einops2 = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
print("einops形式2结果形状:", Y_einops2.shape)
```
矩阵乘法结果形状: torch.size([3, 5, 2])
torch.einsum结果形状: torch.size([3, 5, 2])
einops形式1结果形状: torch.size([3, 5, 2])
einops形式2结果形状: torch.size([3, 5, 2])

```python
import torch
from einops import rearrange, einsum

# 示例（einstein_example2）：使用 einops.rearrange 进行广播操作

# 生成一批图像和一组缩放因子
images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)，表示 64 张 128x128 的 RGB 图像
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)  # (10,)，生成 10 个从 0 到 1 的亮度缩放因子

print("输入张量形状:")
print("images.shape:", images.shape)        # torch.Size([64, 128, 128, 3])
print("dim_by.shape:", dim_by.shape)        # torch.Size([10])

# 方法 1：通过 reshape 和乘法实现广播
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")  # 扩展为 (1, 10, 1, 1, 1)
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")  # (64, 1, 128, 128, 3)
dimmed_images_v1 = images_rearr * dim_value  # 广播相乘
print("\n方法1结果:")
print("dim_value.shape:", dim_value.shape)           # torch.Size([1, 10, 1, 1, 1])
print("images_rearr.shape:", images_rearr.shape)     # torch.Size([64, 1, 128, 128, 3])
print("dimmed_images_v1.shape:", dimmed_images_v1.shape)  # torch.Size([64, 10, 128, 128, 3])

# 方法 2：使用 einsum 一步完成
dimmed_images_v2 = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)

print("\n方法2结果:")
print("dimmed_images_v2.shape:", dimmed_images_v2.shape)  # torch.Size([64, 10, 128, 128, 3])
```
输入张量形状:
images.shape: torch.Size([64, 128, 128, 3])
dim_by.shape: torch.Size([10])

方法1结果:
dim_value.shape: torch.Size([1, 10, 1, 1, 1])
images_rearr.shape: torch.Size([64, 1, 128, 128, 3])
dimmed_images_v1.shape: torch.Size([64, 10, 128, 128, 3])

方法2结果:
dimmed_images_v2.shape: torch.Size([64, 10, 128, 128, 3])

### 3.4 基本构建模块：线性层和嵌入层
#### 3.4.1 参数初始化
正确的参数初始化非常重要。我们采用以下初始化规则：
- **线性层权重**：$W_{ij} \sim \mathcal{N}(0,\sigma^2)$，其中 $\sigma^2=\frac{2}{d_{\text{in}}+d_{\text{out}}}$，并截断至 $\pm 3\sigma$ 范围内。
- **嵌入层（Embeddings）**：每个元素从 $\mathcal{N}(0,1)$ 中采样，并截断到区间 $[-3,3]$。
- **RMSNorm 缩放参数（gains）**：全部初始化为1。

在 PyTorch 中，可以使用 `torch.nn.init.trunc_normal_` 来初始化权重。

#### 3.4.2 线性模块（Linear Module）
线性模块执行矩阵乘法操作。我们实现一个自定义的 Linear 类（继承自 `torch.nn.Module`），不包含偏置项。其前向传播计算为：
$$
y=Wx,
$$

```python
import torch
import torch.nn as nn
input_dim = 16384
hidden_dim = 32
w = nn.Parameter(torch.randn(input_dim,hidden_dim))
x = nn.Parameter(torch.randn(input_dim))
output = x @ w
output
```
tensor([ 285.3951,  210.2427,   34.5771,  -46.3320,   95.1141, -115.6904,
        -154.2949,  -60.1955,   50.3835,  228.9332, -253.0125,  183.3314,
        -226.0321,  -34.4998,   98.7434, -162.0564, -116.2524,  182.7540,
         -36.6848,   -2.1386,   46.6291,  220.2971, -149.2908,  -96.5421,
         -56.3646,    7.7230,    9.7287,  172.0354,  124.5273,  120.1974,
         204.0635, -156.1456], grad_fn=<SqueezeBackward4>)

```python
import numpy as np
w = nn.Parameter(torch.randn(input_dim,hidden_dim) / np.sqrt(input_dim))
output = x @ w
output
```
tensor([-1.1294,  0.0739, -0.0756, -1.2595,  0.1013,  0.4984,  0.4841, -0.3894,
        -0.8523,  0.5237, -0.3436,  0.5862, -0.0189, -0.2259, -0.7464, -0.1600,
        -0.0633,  1.0717, -1.0403, -1.0870, -0.1722, -0.2335, -0.3526, -1.3660,
         1.1142,  1.1098,  0.9295, -0.8007, -0.7902, -0.7361,  1.3478, -0.2251],
       grad_fn=<SqueezeBackward4>)

**问题（1分）实现线性模块**
交付内容：实现一个继承自 torch.nn.Module 的 Linear 类，执行线性变换。你的实现应遵循 PyTorch 内置的 nn.Linear 模块的接口，但不需要支持 bias（偏置）参数。
我们推荐以下接口：
`def __init__(self, in_features, out_features, device=None, dtype=None)`
构造一个线性变换模块。该函数应接受以下参数：
- in_features: int，输入张量的最后一个维度大小
- out_features: int，输出张量的最后一个维度大小
- device: torch.device | None = None，用于存储参数的设备
- dtype: torch.dtype | None = None，参数的数据类型

`def forward(self, x: torch.Tensor) -> torch.Tensor`
对输入 x 应用线性变换并返回结果。
注意事项：
- 必须继承 nn.Module
- 在 `__init__` 中调用父类构造函数（即 `super().__init__()`）
- 创建并存储你的权重参数 W（注意：是 W 而不是 W⊤，出于内存布局的考虑），并将 W 包装为 `nn.Parameter`
- 不能使用 `nn.Linear` 或 `nn.functional.linear`
- 权重初始化：使用 `torch.nn.init.trunc_normal_` 函数进行初始化

代码可见[linear_and_embedding_module.py](hw3/linear_and_embedding_module.py)

#### 3.4.3 嵌入模块（Embedding Module）
如上所述，Transformer 的第一层是一个嵌入层（embedding layer），它将整数形式的 token ID 映射到维度为 d_model 的向量空间。我们将实现一个自定义的 Embedding 类，该类继承自 `torch.nn.Module`（因此你不应使用 nn.Embedding）。其 forward 方法应通过在一个形状为（vocab_size, d_model）的嵌入矩阵中，使用形状为（batch_size, sequence_length）的 torch.LongTensor 类型的 token ID 进行索引，来为每个 token ID 选择对应的嵌入向量。

**问题（1分）实现嵌入模块**
交付内容：实现一个名为 Embedding 的类，该类继承自 `torch.nn.Module`，并执行嵌入查找操作。你的实现应当遵循 PyTorch 内置的 nn.Embedding 模块的接口。我们推荐使用以下接口：

`def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)`
构造一个嵌入模块。该函数应接受以下参数：
- num_embeddings: int，词汇表的大小
- embedding_dim: int，嵌入向量的维度，即 d_model
- device: torch.device | None = None，参数存储的设备
- dtype: torch.dtype | None = None，参数的数据类型

`def forward(self, token_ids: torch.Tensor) -> torch.Tensor`
对给定的 token ID 查找对应的嵌入向量。
注意事项：
- 必须继承 nn.Module 类
- 必须调用父类（超类）的构造函数
- 将嵌入矩阵初始化为 nn.Parameter
- 嵌入矩阵的形状应为 (num_embeddings, embedding_dim)，即 embedding_dim 是最后一个维度
- 不能使用 `nn.Embedding` 或 `nn.functional.embedding`
- 权重初始化请使用 `torch.nn.init.trunc_normal_`（截断正态分布初始化），并使用上述推荐的参数设置

代码可见[linear_and_embedding_module.py](hw3/linear_and_embedding_module.py)

### 3.5 预归一化 Transformer 模块（Pre-Norm Transformer Block）
![](../figures/fig3.png)
每个 Transformer 模块包含两个子层：(1) 多头自注意力机制（Multi-Head Self-Attention）和 (2) 位置级前馈网络（Position-wise Feed-Forward Network）。我们采用**预归一化（pre-norm）结构**：在每个子层之前先进行层归一化。具体来说，若模块输入为$x$，则模块执行如下操作：
1. **自注意力子层**:
$$
y = x + \mathrm{MultiHeadSelfAttention}\!\left(\mathrm{RMSNorm}(x)\right)
$$
2. **前馈网络子层**：
$$
z = y + \mathrm{FFN}\!\left(\mathrm{RMSNorm}(y)\right)
$$

每个残差连接后进入下一个子层。这种预归一化结构（配合 RMSNorm）在深层 Transformer 中有助于提升训练稳定性。
![](../figures/fig4.png)

#### 3.5.1 均方根归一化（Root Mean Square Layer Normalization）
我们使用 **RMSNorm**。给定输入向量 $a \in \mathbb{R}^{d_{\mathrm{model}}}$，RMSNorm 将每个元素除以该向量的均方根，并乘以可学习的缩放因子：

$$
\mathrm{RMSNorm}(a_i) = \frac{a_i}{\mathrm{RMS}(a)} \cdot g_i,
\qquad
\text{其中}\quad
\mathrm{RMS}(a) = \sqrt{\frac{1}{d_{\mathrm{model}}}\sum_{j=1}^{d_{\mathrm{model}}} a_j^2 + \epsilon }.
$$

其中 $g_i$ 是可学习的缩放参数（初始化为 1），$\epsilon$（例如 $1\mathrm{e}{-5}$）用于防止除零错误。  在实现时，应先将输入**提升为 float32 类型**再进行平方运算，以避免上溢，之后再降回原始数据类型。例如：
```python
# 前向传播中的示例框架    
in_dtype = x.dtype    
x = x.to(torch.float32)    
# 执行 RMSNorm（省略具体计算）    
# ...    
result = ...    
# 将结果转换回原始数据类型    
return result.to(in_dtype)
```

**问题（1分）均方根归一化（RMSNorm）**
交付内容：将 RMSNorm 实现为一个 torch.nn.Module。
我们推荐使用以下接口：

`def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)`
构造 RMSNorm 模块。该函数应接受以下参数：
- d_model: int — 模型的隐藏层维度
- eps: float = 1e-5 — 数值稳定用的 epsilon 值
- device: torch.device | None = None — 参数存储的设备
- dtype: torch.dtype | None = None — 参数的数据类型

`def forward(self, x: torch.Tensor) -> torch.Tensor`
处理一个形状为 (batch_size, sequence_length, d_model) 的输入张量，并返回相同形状的张量。
注意：如上所述，在执行归一化之前，请记得先将输入提升（upcast）为 torch.float32 类型（之后再降回原始数据类型）。

代码可见[RMSnorm.py](hw3/RMSnorm.py)

#### 3.5.2 位置级前馈网络（Position-Wise Feed-Forward Network）
在原始的Transformer论文（Vaswani等人[2017]，第3.3节）中，Transformer的前馈网络（Feed-Forward Network, FFN）由两个线性变换组成，中间使用ReLU激活函数（ReLU(x) = max(0, x)）。通常情况下，内部前馈层的维度是输入维度的4倍。

然而，现代语言模型相较于这一原始设计引入了两个主要变化：使用了不同的激活函数，并采用了门控机制。具体来说，我们将实现一种名为“SwiGLU”的激活函数，该函数已被诸如Llama 3 [Grattafiori et al., 2024] 和 Qwen 2.5 [Yang et al., 2024] 等大语言模型（LLM）所采用。SwiGLU结合了SiLU（常被称为Swish）激活函数和一种称为门控线性单元（Gated Linear Unit, GLU）的门控机制。此外，我们还将省略线性层中有时使用的偏置项（bias），这是自PaLM [Chowdhery et al., 2022] 和 LLaMA [Touvron et al., 2023] 以来大多数现代大语言模型的做法。

![](../figures/fig5.png)

SiLU（或称Swish）激活函数 [Hendrycks 和 Gimpel, 2016; Elfwing 等, 2017] 定义如下：
$$
\mathrm{SiLU}(x)=x\cdot\sigma(x)=\frac{x}{1+e^{-x}}
$$

如图所示，SiLU激活函数与ReLU激活函数类似，但在零点处是平滑的。

门控线性单元（GLU）最初由Dauphin等人[2017]提出，其定义为一个经过Sigmoid函数变换的线性变换与另一个线性变换之间的逐元素乘积：
$$
\mathrm{GLU}(x,W_1,W_2)=\sigma(W_1x)\odot W_2x, 
$$

其中 $\odot$ 表示逐元素相乘。门控线性单元被认为可以通过提供一条线性的梯度通路，同时保留非线性能力，从而“减轻深层架构中的梯度消失问题”。

将 SiLU/Swish 激活函数与 GLU 机制结合起来，就得到了 SwiGLU，我们将用它来构建前馈网络：
$$
\mathrm{FFN}(x)=\mathrm{SwiGLU}(x,W_1,W_2,W_3) = W_2\big(\mathrm{SiLU}(W_1x)\odot W_3x\big)
$$

其中 $x\in\mathbb{R}^{d_{\text{model}}}$， $W_1,\;W_3\in\mathbb{R}^{d_{\text{ff}}\times d_{\text{model}}}, W_2\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}}$, 且通常设定 $d_{\text{ff}}=\frac{8}{3}\,d_{\text{model}}$

Shazeer [2020] 首次提出了将SiLU/Swish激活函数与GLU结合的思路，并通过实验表明，在语言建模任务上，SwiGLU的表现优于ReLU以及无门控的SiLU等基线方法。在本作业的后续部分，你也将对SwiGLU和SiLU进行比较。尽管我们已经提到了这些组件的一些启发式理由（相关论文也提供了更多支持性证据），但保持实证视角仍然很重要。Shazeer论文中有一句如今广为流传的话：

“我们并不解释为何这些架构似乎有效；我们将它们的成功归因于——如同其他一切一样——神的仁慈。”

**问题（2分）实现位置级前馈网络**
交付内容：实现 SwiGLU 前馈网络，该网络由 SiLU 激活函数和门控线性单元（GLU）机制组成。
注意：在本题中，出于数值稳定性的考虑，你可以在实现中自由使用 torch.sigmoid。

在实现时，应将前馈层的隐藏维度 $d_{\text{ff}}$ 设置为大约 $\frac{8}{3}\,d_{\text{model}}$ ，同时确保该维度是 64 的倍数，以便更好地利用硬件计算资源（如 GPU 的并行计算能力）。

代码可见[SwiGLU.py](hw3/SwiGLU.py)

#### 3.5.3 旋转位置编码（RoPE）
如果可以对查询（Q）或键（K）向量使用合适的位置编码方式，那么注意力打分（点积）就能写成只依赖内容 $xx_m,xx_n$ 和相对位置 $m-n$ 的函数了，如下所示：
$$
\langle f_q(xx_m,m),\, f_k(xx_n,n)\rangle = g(xx_m,xx_n,\, m-n)
$$

为了编码位置信息，我们使用**旋转位置编码（Rotary Position Embedding, RoPE）**。对于每个位置 $i$ 和一个 $d$ 维向量 $q^{(i)}$，我们对每对维度 $(2k, 2k-1)$ 施加一个角度为 $\theta_{i,k}$ 的旋转。令 $k = 1,\ldots,d/2$，定义：
$$
\theta_{i,k} = i \cdot \frac{\Theta}{10000^{2k/d}}
$$

其中 $\Theta$ 为常数。在维度 $(2k,2k-1)$ 上的 $2 \times 2$ 旋转矩阵为：
$$
R_{i,k}=
\begin{pmatrix}
\cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\
\sin(\theta_{i,k}) & \cos(\theta_{i,k})
\end{pmatrix}
$$

所有这些 $2 \times 2$ 块构成一个完整的 $d \times d$ 分块对角旋转矩阵 $R_i$。我们将 $R_i$ 应用于查询（Q）或键（K）向量（不作用于值向量 V）。实践中，我们可以通过预先计算正弦/余弦表（注册为 buffer，而非参数）来实现 RoPE，表的大小为 $(max\_seq\_len,d)$。前向传播时，根据实际序列长度切片对应的 sin/cos 值并应用。相同的旋转在所有注意力头之间共享（将头维度视为旋转的批处理维度）。

**问题（2 分）实现 RoPE**
交付内容：实现一个名为 RotaryPositionalEmbedding 的类，将 RoPE（旋转位置编码）应用到输入张量上。
推荐的接口如下：
`def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)`
- 构造 RoPE 模块，并在需要时创建缓存（buffers）。
- theta: RoPE 中的 $\Theta$ 值（控制旋转角度的频率基底）。
- d_k: 查询（query）和键（key）向量的维度。
- max_seq_len: 输入序列的最大长度。
- device: 存储缓存张量的设备（torch.device 或 None）。

`def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor`
- 处理形状为 (..., seq_len, d_k) 的输入张量 x，返回相同形状的输出张量。
- 应支持任意数量的批处理维度（batch dimensions）。
- 假设 token_positions 是一个形状为 (..., seq_len) 的张量，表示 x 在序列维度上的各个 token 的位置索引。
- 你应该使用 token_positions 来从预先计算好的 cos 和 sin 张量中沿序列维度进行索引（切片）。

代码可见[rope.py](hw3/rope.py)

#### 3.5.4 缩放点积注意力（Scaled Dot-Product Attention）
将注意力（Attention）操作从数学上定义如下：
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

其中 $Q\in\mathbb{R}^{n\times d_k},K\in\mathbb{R}^{m\times d_k},V\in\mathbb{R}^{m\times d_v}$。这里的 $Q$、$K$ 和 $V$ 都是该操作的输入,注意，它们不是可学习的参数。

**掩码（Masking）**：有时我们希望对注意力操作的输出进行掩码处理。掩码矩阵的形状为 $M\in\{\mathrm{True},\mathrm{False}\}^{n\times m}$，其每一行 $i$ 表示第 $i$ 个查询（query）应当关注哪些键（key）。通常（但稍显反直觉地），掩码中位置 $(i,j)$ 处的值为 True 表示查询 $i$ **可以关注**键 $j$，而 False 表示**不能关注**。换句话说，信息只在值为 True 的 $(i,j)$ 位置流动。例如，考虑一个 $1 \times 3$ 的掩码矩阵 $[[True,True,False]]$，则唯一的查询向量仅关注前两个键。

从计算角度看，使用掩码比在子序列上单独计算注意力更高效。我们可以通过在 softmax 前的注意力分数矩阵 $\frac{QK^{\top}}{\sqrt{d_k}}$ 上，将掩码为 False 的位置加上 $-\infty$ 来实现这一效果（实际实现中通常用极小的负数如 -1e9 代替 $-\infty$）。这样，softmax 会将这些位置的权重置为接近零，从而屏蔽对应的信息流动。

**问题（1 分）实现 softmax**
交付内容：编写一个函数，对张量执行 softmax 操作。你的函数应接受两个参数：一个张量和一个维度 i，并在输入张量的第 i 维上应用 softmax。输出张量应与输入张量具有相同的形状，但其第 i 维将变为归一化的概率分布。

为了防止数值不稳定问题，你需要使用“减去最大值”的技巧：在对第 i 维进行指数运算前，先从该维的每个元素中减去该维的最大值。

代码可见[softmax.py](hw3/softmax.py)

**问题（5 分）实现缩放点积注意力**
交付内容：实现缩放点积注意力函数。你的实现应支持形状为 (batch_size, ..., seq_len, d_k) 的查询（Q）和键（K），以及形状为 (batch_size, ..., seq_len, d_v) 的值（V），其中 ... 表示任意数量的批处理类维度（如果存在）。函数应返回形状为 (batch_size, ..., d_v) 的输出。关于批处理类维度的讨论，详见第 3.3 节。

你的实现还应支持一个可选的、用户提供的布尔类型掩码（mask），其形状为 (seq_len, seq_len)。对于掩码值为 True 的位置，其对应的注意力概率应正常计算并归一化（总和为 1）；而对于掩码值为 False 的位置，其注意力概率应为 0（即被屏蔽）。

![](../figures/fig6.png)

代码可见[scaled_dot_product_attention.py](hw3/scaled_dot_product_attention.py)

#### 3.5.5 因果多头自注意力（Causal Multi-Head Self-Attention）
我们将实现如 Vaswani 等人 [2017] 论文第 3.2.2 节所述的**多头自注意力机制**。回顾一下，从数学上讲，多头注意力的操作定义如下：
$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)
$$

其中
$$
\mathrm{head}_i=\mathrm{Attention}(Q_i,K_i,V_i)
$$

这里的 $Q_i,K_i,V_i$ 分别是查询 $Q$、键 $K$ 和值 $V$ 在嵌入维度上的第 $i \in \{1,\dots,h\}$ 个切片，每个切片大小为 $d_k$ 或 $d_v$。其中 $\mathrm{Attention}$ 是 §3.5.4 中定义的**缩放点积注意力**操作。

由此可得**多头自注意力**的表达式：
$$
\mathrm{MultiHeadSelfAttention}(x)=W_O\cdot \mathrm{MultiHead}(W_Qx,\;W_Kx,\;W_Vx)
$$

其中可学习参数为：
- $W_Q\in\mathbb{R}^{h d_k\times d_{\text{model}}}$
- $W_K\in\mathbb{R}^{h d_k\times d_{\text{model}}}$
- $W_V\in\mathbb{R}^{h d_v\times d_{\text{model}}}$
- $W_O\in\mathbb{R}^{d_{\text{model}}\times h d_v}$ 

由于在多头注意力中 $Q,K,V$ 会被沿输出维度切分为 $h$ 个头，我们可以将 $W_Q,W_K,W_V$ 理解为每个注意力头各自对应的投影矩阵。当实现正确时，你应该仅通过**三次矩阵乘法**就完成所有查询、键和值的投影。

**因果掩码（Causal Masking）**
你的实现应防止模型关注序列中的未来 token。换句话说，如果输入序列为 $t_1,\ldots,t_n$，而我们正在计算前缀 $t_1,\ldots,t_i$（其中 $i<n$ ）的下一个词预测时，模型不应能访问位置 $t_{i+1},\ldots,t_n$ 的表示。因为在推理生成文本时，模型无法提前看到未来的 token；若允许访问，会泄露真实下一个词的信息，从而让语言建模任务变得平凡。

最朴素的方法是对序列的每一个前缀单独运行一次注意力（共 $n$ 次），但我们采用**因果注意力掩码**来高效解决这个问题：它允许位置 $i$ 只关注所有满足 $j \leq i$ 的位置 $j$。你可以使用 torch.triu 或广播的索引比较来构造这样的上三角掩码（上三角部分为 False），并利用你在 §3.5.4 中实现的缩放点积注意力已支持掩码这一特性。

**应用 RoPE（旋转位置编码）**
RoPE 应**仅应用于查询（Q）和键（K）向量，不应用于值（V）向量**。此外，在处理多头结构时，**头维度应被视为批处理维度**，因为每个注意力头是独立计算注意力的。这意味着相同的 RoPE 旋转操作应分别应用于每个头的查询和键向量——即对每个头独立地应用相同的位置旋转方式（共享旋转频率，但按头独立作用于其对应的 Q 和 K）。

**问题（5 分）实现因果多头自注意力**
交付内容：将因果多头自注意力实现为一个 torch.nn.Module 模块。你的实现应至少接受以下参数：
- d_model: int，Transformer 模块输入的特征维度。
- num_heads: int，多头自注意力中使用的注意力头数量。

按照 Vaswani 等人 [2017] 的设定，令每个头的键 $d_k$ 和值 $d_v$ 的维度为：
$$
d_k = d_v = \frac{d_{model}}{h}
$$

你的实现应满足：
- 使用因果掩码（causal mask），防止每个位置关注未来 token；
- 支持批量输入和任意序列长度；
- 在查询（Q）和键（K）上应用 RoPE 位置编码（不应用于值 V）；
- 通过线性投影生成 Q、K、V，并正确分割为多个头；
- 最终输出经过输出投影，保持输出维度为 d_model。

不带 RoPE 位置编码的代码可见[causal_multi_head_attention.py](hw3/causal_multi_head_attention.py)，带 RoPE 位置编码的代码可见[causal_multi_head_attention_with_rope.py](hw3/causal_multi_head_attention_with_rope.py)

### 3.6 完整的 Transformer 语言模型
让我们开始构建 Transformer 块。一个 Transformer 块包含两个“子层”，一个用于多头自注意力（Multi-Head Self-Attention），另一个用于前馈网络（Feed-Forward Network）。在每个子层中，我们首先进行 RMSNorm 归一化，然后执行主要操作（MHA 或 FF），最后将结果通过残差连接加到原始输入上。

具体来说，Transformer 块的前半部分（即第一个“子层”）应实现以下更新过程，将输入 x 转换为输出 y：
$$
y=x+\mathrm{MultiHeadSelfAttention}(\mathrm{RMSNorm}(x))
$$

**问题（3分）实现Transformer块**
请按照第3.5节的描述，实现预归一化（pre-norm）的 Transformer 块。你的 Transformer 块至少应支持以下参数：
- d_model: int，Transformer块输入的特征维度。
- num_heads: int，多头自注意力机制中使用的注意力头数量。
- d_ff: int，位置前馈网络（feed-forward）中间层的维度。

代码可见[transformer_block.py](hw3/transformer_block.py)

现在我们将各个模块组合起来，遵循图1中的高层结构示意图。根据第3.1.1节中对嵌入（embedding）的描述，先生成输入嵌入，然后将其依次输入到 num_layers 个Transformer块中，最后将输出传入输出层，从而得到词汇表上的概率分布。

**问题（3分）实现Transformer语言模型**
现在是时候将所有部分整合起来了！请根据第3.1节的描述并参考图1的结构，实现 Transformer 语言模型。你的实现至少应包含之前提到的所有Transformer块的构造参数，以及以下额外参数：
- vocab_size: int，词汇表的大小，用于确定词元（token）嵌入矩阵的维度。
- context_length: int，最大上下文长度，用于确定位置嵌入矩阵的维度。
- num_layers: int，使用的Transformer块的数量。

交付内容：一个能够通过上述测试的 TransformerLM 模块。

代码可见[run_transformer_lm.py](hw3/run_transformer_lm.py)

**资源核算（Resource Accounting）**：
了解Transformer各个部分在计算和内存上的消耗是非常有用的。接下来，我们将逐步进行一些基本的“FLOPs 核算”（即计算量统计）。Transformer 中绝大多数的浮点运算（FLOPs）来自矩阵乘法，因此我们的核心方法很简单：
1. 列出 Transformer 前向传播（forward pass）中所有的矩阵乘法操作。
2. 将每个矩阵乘法转换为所需的 FLOPs（浮点运算次数）。

对于第二步，以下规则非常有用：

**规则**：给定矩阵 $A\in\mathbb{R}^{m \times n}$ 和 $B\in\mathbb{R}^{n \times p}$，矩阵乘积 $AB$ 需要 $2mnp$ 次FLOPs。

**解释**：
因为乘积结果中的每个元素$(AB)[i,j]=A[i,:] \cdot B[:,j]$是一个向量点积，该点积包含 $n$ 次乘法和 $n$ 次加法，共 $2n$ 次FLOPs。而 $AB$ 一共有 $m \times p$ 个元素，因此总计算量为 $(2n) \times (m\times p) = 2mnp \mathrm{FLOPS}$

在你进行下一个问题之前，建议先仔细检查你实现的 Transformer 块和 Transformer 语言模型的每个组件，列出其中所有的矩阵乘法操作，并计算它们各自对应的 FLOPs 开销。这将帮助你更准确地分析模型的计算复杂度。

**问题（5分）TransformerLM 资源核算**
(a) 考虑 GPT-2 XL 模型，其配置如下：
- vocab_size: 50,257
- context_length: 1,024
- num_layers: 48
- d_model: 1,600
- num_heads: 25
- d_ff: 6,400

假设我们使用该配置构建模型，该模型共有多少可训练参数？若每个参数以单精度浮点数（32位）存储，仅加载该模型需要多少内存？
交付内容：一到两句话的简要回答。

回答：
Embedding 参数计算
- token embedding：vocab_size $\times$ d_model = 50,257 $\times$ 1600 = 80,411,200
- positional embedding：context_length $\times$ d_model = 1,024 $\times$ 1600 = 1,638,400

合计：82,049,600
每个 Transformer Block（每层）
- Self-Attention：
  - QKV 投影：权重 $d \times 3d=3d^2$, bias $3d$
  - 输出投影：权重 $d \times d=d^2$, bias $d$
  合计：权重 $4d^2$, bias $4d$
- MLP：
  - $d \rightarrow d_{ff}$：权重 $d \cdot d_{ff}$， bias $d_{ff}$
  - $d_{ff} \rightarrow d$：权重 $d_{ff} \cdot d$， bias $d$
  合计：权重 $2d_{ff} \cdot d$, bias $d_{ff}+d$
- 2 个 LayerNorm：每个有 weight+bias 共 $2d$，两个共 $4d$

因此每层参数：
$$
4d^2 + 2d \cdot d_{ff} + (4d + 4d + d_{ff}+d) = 4d^2 + 2d \cdot d_{ff} + (9d + d_{ff})
$$

代入 $d=1600,d_{ff}=6400$：每层参数量为30,740,800。48层共1,475,558,400。最后 LayerNorm 参数量为 $2d=3200$。
总参数量为1,557,611,200。

每个参数 32bit = 4 bytes：
$$
1,557,611,200 \times 4=6,230,444,800 bytes
$$

约为6.23GB。

---

(b) 列出完成一次 GPT-2 XL 规模模型前向传播所需的所有矩阵乘法操作。这些矩阵乘法总共需要多少次 FLOPs？假设输入序列长度为 context_length 个 token。
交付内容：一份带说明的矩阵乘法列表，以及所需的总 FLOPs 数量。

**单层（1 个 Transformer block）的矩阵乘法清单与 FLOPs**

| 模块              | 矩阵乘法（形状）                                |  次数 | FLOPs 公式                             |       数值 FLOPs |
| --------------- | --------------------------------------- | --: | ------------------------------------ | -------------: |
| QKV 投影          | $(T\times d)\cdot(d\times 3d)$     |   1 | $2T d (3d)=6Td^2$     | 15,728,640,000 |
| 注意力打分 $QK^\top$ | 每头：$(T\times d_h)\cdot(d_h\times T)$    | $h$ | $h\cdot 2T d_h T = 2T^2(hd_h)=2T^2d$ |  3,355,443,200 |
| 注意力加权 $PV$     | 每头：$(T\times T)\cdot(T\times d_h)$     | $h$ | $h\cdot 2T T d_h=2T^2d$              |  3,355,443,200 |
| 输出投影 $WO$       | $(T\times d)\cdot(d\times d)$           |   1 | $2Td^2$                             |  5,242,880,000 |
| FFN 上投影         | $(T\times d)\cdot(d\times d_{ff})$      |   1 | $2Tdd_{ff}$                         | 20,971,520,000 |
| FFN 下投影         | $(T\times d_{ff})\cdot(d_{ff}\times d)$ |   1 | $2T d_{ff} d$                        | 20,971,520,000 |

**单层合计：**
$6Td^2 + 4T^2d + 4Tdd_{ff}$
数值：**69,625,446,400 FLOPs**（约 6.96e10）

**48 层总 FLOPs（只算 block 内矩阵乘法）**

把上面单层乘以 (L=48)：

* QKV：$15.72864\text{B}\times 48 = 754.97472\text{B}$
* $QK^\top$：$3.3554432\text{B}\times 48 = 161.0612736\text{B}$
* $PV$：同上 $=161.0612736\text{B}$
* 输出投影：$5.24288\text{B}\times 48 = 251.65824\text{B}$
* FFN 上：$20.97152\text{B}\times 48 = 1,006.63296\text{B}$
* FFN 下：同上 $=1,006.63296\text{B}$

**48 层 block 内矩阵乘法总计：**
$3,342,021,427,200\ \text{FLOPs}\ \approx 3.342\times 10^{12}$

**最终词表 logits 投影（LM head）**
（权重是否与 embedding 共享不影响 FLOPs，只要要算 logits 就要做这次乘法）
$(T\times d)\cdot(d\times V)\Rightarrow 2T d V$
数值：$2\cdot 1024\cdot 1600\cdot 50257 = 164,682,137,600\ \text{FLOPs}$

**一次完整前向传播（矩阵乘法部分）总 FLOPs**
$\text{Total} = (48\ \text{层 block}) + (\text{LM head})$
$= 3,342,021,427,200 + 164,682,137,600 = 3,506,703,564,800\ \text{FLOPs}$

**最终答案：约 (3.51\times 10^{12}) FLOPs（≈ 3.51 TFLOPs）**（batch=1、只计矩阵乘法）。

---

(c) 根据上述分析，模型的哪些部分消耗的 FLOPs 最多？
交付内容：一到两句话的简要回答。

结论：GPT-2 XL 这种结构里，FLOPs 最大头几乎都在 FFN（MLP）上，约占 60% 左右；注意力相关（含 QKV、WO、QK、PV）合起来约 40%。

---

(d) 使用以下模型配置重复上述分析：
- GPT-2 small：12 层，d_model = 768，12 个注意力头
- GPT-2 medium：24 层，d_model = 1024，16 个注意力头
- GPT-2 large：36 层，d_model = 1280，20 个注意力头
  
随着模型规模增大，TransformerLM 的各个部分在总 FLOPs 中所占比例是上升还是下降？
交付内容：对每个模型，提供各组件的 FLOPs 分解（占前向传播总 FLOPs 的比例），并用一到两句话描述模型规模变化如何影响各组件的 FLOPs 占比。

回答：
越大的 GPT-2，计算越来越被“$d^2$项”（尤其 FFN）主导；而只线性随 (d) 增长的部分（注意力的 $T^2d$ 和 logits 的 $TdV$）相对变“更便宜”。

---

(e) 将 GPT-2 XL 的上下文长度增加到 16,384。一次前向传播的总 FLOPs 如何变化？各模型组件的 FLOPs 相对贡献有何变化？
交付内容：一到两句话的简要回答。

回答：
把 GPT-2 XL 的 $T$ 从 1,024 提到 16,384（×16）后，一次前向的矩阵乘法 FLOPs 从约 $3.51\times10^{12}$ 增到约 $1.33\times10^{14}$，约 **×38**（因为注意力里的 $QK^\top$ 与 $PV$ 是 $O(T^2)$）。相对贡献会从原先 **FFN+投影占大头** 转为 **注意力的 $QK^\top/PV$ 占主导**：它从约 **9%** 跃升到约 **62%**，而 FFN 从约 **57%** 降到约 **24%**（LM head 约 **4.7%→2%**）。


## 4 训练Transformer LM
我们现在已经有了通过分词器预处理数据和模型（Transformer）的步骤。接下来需要完成支持训练的所有代码，主要包括以下部分：
- 损失函数：需要定义损失函数（交叉熵）。
- 优化器：需要定义用于最小化该损失的优化器（AdamW）。
- 训练循环：需要构建所有支持性的基础设施，包括加载数据、保存检查点以及管理训练过程。

### 4.1 交叉熵损失
回顾一下，Transformer 语言模型为每个长度为 $m+1$ 的序列 $x$ 和每个位置 $i=1, \ldots, m$ 定义了一个分布 $p_{\theta}\!\left(x_{i+1}\mid x_{1:i}\right)$，意思是看见前 $i$ 个 token 去预测第 $i+1$ 个 token 是什么。给定一个由长度为 $m$ 的序列组成的训练集 $D$，我们定义标准的交叉熵（负对数似然）损失函数：
$$
\ell(\theta; D)
= \frac{1}{|D|\,m}
\sum_{x\in D}\sum_{i=1}^{m}
-\log p_{\theta}\!\left(x_{i+1}\mid x_{1:i}\right).
$$

其中 $\sum_{i=1}^{m}$ 表示一条序列里，每个位置都要预测一次下一个词（总共 $m$ 次）。$-\log p_{\theta}(\cdot)$ 表示如果模型给“正确答案 token”的概率越大，$−log$ 就越小，损失也就越小。 $\frac{1}{|D|\,m}$ 表示对数据条数和序列位置数做平均。注意：Transformer 的一次前向传播可以得到所有 $i=1, \ldots, m$ 对应的 $p_{\theta}\!\left(x_{i+1}\mid x_{1:i}\right)$。 

具体而言，Transformer 为每个位置 $i$ 计算 logits 向量 $o_i \in \mathbb{R}^{\text{vocab\_size}}$，从而得到：
$$
p\!\left(x_{i+1}\mid x_{1:i}\right)
= \mathrm{softmax}(o_i)[x_{i+1}]
= \frac{\exp\!\big(o_i[x_{i+1}]\big)}
{\sum_{a=1}^{\text{vocab\_size}} \exp\!\big(o_i[a]\big)}.
$$

可以把 $o_i[a]$ 理解为“位置 $i$ 预测下一个 token 是词表里第 $a$ 个 token 的打分”（未归一化）。交叉熵损失通常定义为关于 logits 向量 $o_i \in \mathbb{R}^{\text{vocab\_size}}$ 和目标值 $x_{i+1}$。

实现交叉熵损失时需要特别注意数值稳定性问题，这一点与 softmax 的实现类似。

**问题（cross_entropy）：实现交叉熵损失**
交付内容：编写一个函数来计算交叉熵损失，该函数接收预测的 logits $o_i$和目标值 $x_{i+1}$，并计算交叉熵 $\ell_i = −log(softmax(o_i)[x_{i+1}])$。你的函数应满足以下要求：
- 减去最大值以保证数值稳定性。
- 尽可能约去 log 和 exp 运算，避免数值溢出或下溢。
- 能够处理任意的批量（batch）维度，并对 batch 维度求平均后返回结果。

与第 3.3 节一样，我们假设批量相关的维度始终位于词汇表维度（vocab_size）之前。

代码可见[cross_entropy.py](hw4/cross_entropy.py)

**困惑度（Perplexity）**
交叉熵足以用于训练，但在评估模型时，我们还希望报告困惑度，主要是用来评价语言模型“预测下一个词有多难/有多不确定”的指标。对于一个长度为 $m$ 的序列，若其对应的交叉熵损失分别为 $\ell_1,\ldots,\ell_m$：
$$
perplexity=\mathrm{exp}\left(\frac{1}{m}\sum_{i=1}^{m}\ell_i \right)
$$

在很多情况下可以把 PPL 理解成：模型在每一步相当于在 $K$ 个选项里“平均在猜”，这个 $K$ 就是困惑度。举例：
- 如果模型每步都像在 2 个词里均匀猜（正确词概率约 0.5）
  $\ell \approx -\log 0.5 = 0.693$，ppl $=e^{0.693}\approx 2$
- 如果每步正确词概率约 0.1
  $\ell\approx 2.302)，ppl (=e^{2.302}\approx 10$

所以 ppl 越大，说明模型越“困惑”，每一步要在更多可能里徘徊。


### 4.2 随机梯度下降优化器
现在我们有了损失函数，接下来将开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。我们从随机初始化的参数 $\theta_0$ 开始。然后对于每一步 $t=0,\ldots,T-1$，执行以下更新：
$$
\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t),
$$

其中 $B_t$ 是从数据集 $D$ 中采样的随机批量数据，学习率 $α_t$ 和批量大小 $|Bt|$ 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD
要实现我们的优化器，我们将继承 PyTorch 的 `torch.optim.Optimizer` 类。一个 `Optimizer` 子类必须实现两个方法：
- `def __init__(self, params, ...)` 应初始化优化器。`params` 将是需要优化的参数集合（或参数组，如果用户想为模型的不同部分使用不同的超参数，例如不同的学习率）。确保将 params 传递给基类的 `__init__` 方法，该方法会将这些参数存储起来以供后续步骤使用。你可以根据优化器的需求添加额外的参数（例如，学习率是一个常见参数），并将它们**作为字典传递给基类构造函数**，字典中的键是你为这些参数选择的名称（字符串）。
- `def step(self)` 应执行一次参数更新。在训练循环中，这个方法会在反向传播后被调用，因此你可以访问到上一批数据的梯度。该方法应遍历每个参数张量 `p` 并就地修改它们，即设置 `p.data`，它保存了与该参数相关的张量，基于梯度 `p.grad`（如果存在的话）——该梯度是相对于该参数的损失梯度张量。

PyTorch 优化器 API 有一些微妙之处，所以用一个例子来解释会更简单。为了使示例更丰富，我们将实现 SGD 的一个变体，其中学习率随训练过程衰减，从一个初始学习率 $\alpha$ 开始，随着时间推移逐步缩小步长：
$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t; B_t),
$$

让我们看看如何将这个版本的 SGD 实现为 PyTorch Optimizer：
- 在 `__init__` 中，我们将参数以及默认的超参数传递给基类构造函数（参数可能以组的形式出现，每组具有不同的超参数）。如果参数只是单个 `torch.nn.Parameter` 对象的集合，基类构造函数将创建一个单独的组并将其分配给默认超参数。
- 在 `step` 中，我们迭代每个参数组，然后对组内的每个参数应用上述公式。在这里，我们将迭代次数作为状态与每个参数关联：首先读取此值，将其用于梯度更新，然后更新它。API 规定用户可能会传入一个可调用的 closure 来重新计算优化器步骤之前的损失。我们不需要为此优化器使用它，但为了符合 API，我们将其添加进来。

要查看其工作原理，我们可以使用以下最小化训练循环示例：
```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        """
        简易SGD优化器，学习率按 sqrt(t+1) 衰减
        把要优化的参数收起来，并存好默认超参数（比如学习率 lr）
        把参数组织成 self.param_groups（参数组）
        把默认超参数也放进每个 group 里（所以后面能 group["lr"]）
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """
        执行一次参数更新
        支持可选的闭包函数用于重新计算模型
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # 当前参数组的学习率

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # 获取参数状态（用于记录迭代次数）
                step_count = state.get("step_count", 0)  # 读取当前步数，初始为0
                grad = p.grad.data  # 梯度数据

                # 更新参数：权重衰减因子为 1/sqrt(step_count + 1)
                p.data -= lr / math.sqrt(step_count + 1) * grad

                # 步数递增并保存回状态
                state["step_count"] = step_count + 1

        return loss


# ------------------- 测试代码 -------------------
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))  # 初始化可学习参数
opt = SGD([weights], lr=1)  # 创建优化器

for t in range(100):
    opt.zero_grad()              # 清零梯度
    loss = (weights**2).mean()   # 定义目标：最小化权重平方均值
    print(loss.cpu().item())     # 打印当前损失
    loss.backward()              # 反向传播计算梯度
    opt.step()                   # 更新参数
```

**问题（learning_rate_tuning）：调整学习率（1分）**
我们将看到，对训练影响最大的超参数之一就是学习率。让我们在之前的SGD示例中实践这一点：将学习率分别设置为1e1、1e2和1e3，仅运行10次训练迭代。对于每个学习率，损失值如何变化？它是下降得更快、更慢，还是发散（即在训练过程中损失值增加）？
提交内容：用一到两句话描述你观察到的现象。

在这个二次损失的测试里，**lr=1e1** 时 loss 稳定、逐步下降；**lr=1e2** 时一开始会出现一次“翻转”导致 loss 基本不变，但随后会在几步内迅速下降到接近 0；而 **lr=1e3** 时更新步长过大，loss 很快爆炸式增大并发散。

### 4.3 AdamW
现代语言模型通常使用比SGD更复杂的优化器进行训练。最近广泛使用的优化器大多源自 Adam 优化器 [Kingma 和 Ba, 2015]。我们将使用 AdamW [Loshchilov 和 Hutter, 2019]，这是近期研究中广泛采用的一种优化器。AdamW 对 Adam 进行了改进，通过引入权重衰减（在每次迭代中将参数向0拉近）来提升正则化效果，并且该权重衰减与梯度更新解耦。我们将按照 Loshchilov 和 Hutter [2019]论文中算法2的描述来实现 AdamW。

AdamW 是有状态的：对每个参数，它会维护其一阶和二阶矩的滑动估计值。因此，AdamW 以额外的内存消耗为代价，换取了更好的训练稳定性和收敛性。除了学习率 $\alpha$ 外，AdamW 还包含一对控制矩估计更新的超参数 ($\beta_1, \beta_2$)，以及一个权重衰减率 $\lambda$。通常情况下，($\beta_1, \beta_2$) 设为 (0.9, 0.999)，但像 LLaMA [Touvron 等, 2023] 和 GPT-3 [Brown 等, 2020] 这样的大语言模型通常使用 (0.9, 0.95)。该算法可表述如下，其中 $\epsilon$ 是一个很小的值（例如 $10^{-8}$），用于在 $v$ 出现极小值时提升数值稳定性：
![](../figures/fig7.png)
请注意，$t$ 从 1 开始。现在，您将实现此优化器。

**问题（adamw）：实现 AdamW（2 分）**
提交要求：将 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。你的类在 `__init__` 中应接收学习率 $\alpha$，以及超参数 $\beta_1$、$\beta_2$、$\epsilon$ 和 $\lambda$。为了帮助你维护状态，基类 Optimizer 提供了一个字典 `self.state`，它将 `nn.Parameter` 对象映射到一个字典，用于存储该参数所需的任何信息（对 AdamW 而言，即一阶和二阶矩估计值）。

代码可见[adamw.py](hw4/adamw.py)

**问题（adamwAccounting）：使用 AdamW 训练的资源核算（2 分）**
让我们计算运行 AdamW 所需的内存和计算量。假设所有张量均使用 float32（每个元素占 4 字节）。

---
(a) 运行 AdamW 所需的峰值内存是多少？请根据参数、激活值（activations）、梯度和优化器状态的内存使用情况分解回答。用 batch_size 和模型超参数（vocab_size、context_length、num_layers、d_model、num_heads）表示你的答案。假设 d_ff = 4 × d_model。为简化激活值的内存计算，仅考虑以下组件：
- Transformer 块：
  - RMSNorm(s)
  - 多头自注意力子层：QKV 投影、QᵀK 矩阵乘法、softmax、加权求和（value 加权和）、输出投影
  - 位置前馈网络（FFN）：W1 矩阵乘法、SiLU 激活、W2 矩阵乘法
- 最终的 RMSNorm
- 输出嵌入（output embedding）
- logits 上的交叉熵损失

提交要求：分别给出参数、激活值、梯度和优化器状态的代数表达式，以及总内存的表达式。

---
(b) 将你的答案代入一个 GPT-2 XL 规模的模型，得到仅依赖于 batch_size 的表达式。在不超过 80GB 内存的前提下，最大 batch_size 是多少？
提交要求：形如 a·batch_size + b 的表达式（其中 a、b 为具体数值），以及最大 batch_size 的数值。

---
(c) 运行一步 AdamW 需要多少次浮点运算（FLOPs）？
提交要求：一个代数表达式，并附简要说明。

---
(d) 模型 FLOPs 利用率（Model FLOPs Utilization, MFU）定义为实际吞吐量（每秒处理的 token 数）与硬件理论峰值 FLOPs 吞吐量的比值 [Chowdhery et al., 2022]。一块 NVIDIA A100 GPU 的 float32 峰值性能为 19.5 teraFLOP/s。假设你能达到 50% 的 MFU，训练一个 GPT-2 XL 模型，共 400K 步，每步 batch_size = 1024，在单块 A100 上需要多长时间？根据 Kaplan et al. [2020] 和 Hoffmann et al. [2022] 的假设，反向传播的 FLOPs 是前向传播的两倍。

提交要求：训练所需的天数，并附简要说明。

### 4.4 学习率调度
导致损失最快减少的学习率值在训练过程中常常会发生变化。在训练 Transformer 模型时，通常采用学习率调度策略：先使用较大的学习率，在训练初期进行更快速的更新，再随着模型的训练逐渐将其衰减至较小的值。在本作业中，我们将实现用于训练 LLaMA 的余弦退火调度策略 [Touvron et al., 2023]。

调度器本质上是一个函数，它接收当前步数 $t$ 和其他相关参数（如初始学习率和最终学习率），并返回在第 $t$ 步梯度更新时应使用的学习率。最简单的调度器是常函数，它会针对任意 $t$ 返回相同的学习率。

余弦退火学习率调度器需要以下输入：(i) 当前迭代次数 $t$，(ii) 最大学习率 $\alpha_{max}$，(iii) 最小（最终）学习率 $\alpha_{min}$，(iv) 预热迭代次数 $T_w$，以及 (v) 余弦退火迭代次数 $T_c$。第 $t$ 次迭代时的学习率定义如下：
- **预热阶段**：若 $t < T_w$，则 $\alpha_t = \frac{t}{T_w}\,\alpha_{\max}.$
- **余弦退火阶段**：若 $T_w \le t \le T_c$，则 $\alpha_t = \alpha_{\min} + \frac{1}{2}\left(1+\cos\!\left(\frac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{\max}-\alpha_{\min}).$

- **退火后阶段**：若 $t > T_c$，则 $\alpha_t = \alpha_{\min}.$

**问题（learning_rate_schedule）：实现带预热的余弦学习率调度**
编写一个函数，该函数接收参数 $t$（当前训练步数）、$\alpha_{max}$（最大学习率）、$\alpha_{min}$（最小学习率）、$T_w$（预热步数）和 $T_c$（总训练步数），并根据上述定义的学习率调度策略返回当前步数对应的学习率 $α_t$。

代码可见[lr_cosine_shedule.py](hw4/lr_cosine_shedule.py)


### 4.5 梯度裁剪
在训练过程中，我们有时会遇到产生较大梯度的训练样本，这可能会导致训练不稳定。为缓解这一问题，实践中常用的一种技术是梯度裁剪（gradient clipping）。其核心思想是在每次反向传播后，对梯度的范数施加一个上限。

给定所有参数的梯度 $g$，我们计算其 $\ell_2$-范数$\lVert g\rVert_2$。如果该范数小于最大值 $M$，则保持 $g$ 不变；否则，我们将 $g$ 缩放因子为 $\frac{M}{\lVert g\rVert_2+\epsilon}$（其中添加了一个很小的 $\epsilon$ ，如 $10^{−6}$，以确保数值稳定性）。注意，缩放后的范数将略小于 $M$。

**问题（gradient_clipping）：实现梯度裁剪（1 分）**
编写一个函数来实现梯度裁剪。你的函数应该接收一组参数和一个最大 $\ell_2$-范数，并就地修改每个参数的梯度。使用 $\epsilon=10^{−6}$（这是 PyTorch 的默认值）。

代码可见[gradient_clip.py](hw4/gradient_clip.py)


## 5 训练与生成
我们来到了 Training loop 和 Generating text 部分，我们将最终整合迄今为止构建的主要组件：分词后的数据、模型和优化器。

### 5.1 数据加载（Data Loader）
分词后的数据（例如你在 tokenizer_experiments 中准备的数据）是一个单一的 token 序列 $x = (x_1, \ldots, x_n)$。尽管原始数据可能由多个独立的文档组成（例如不同的网页或源代码文件），一种常见的做法是将所有这些文档连接成一个单一的 token 序列，并在它们之间添加一个分隔符（例如 $\text{<|endoftext|>}$ token）。

数据加载器会将这个长序列转换为一批批的数据流，其中每个批次包含 $B$ 个长度为 $m$ 的序列，以及对应的下一个 token 序列（长度也为 $m$）。例如，当 $B$ = 1，$m$ = 3 时，$([x_2, x_3, x_4], [x_3, x_4, x_5])$ 就是一个可能的批次。

以这种方式加载数据在多个方面简化了训练过程。首先，任何满足 $1 \le i < n − m$ 的位置 $i$ 都可以生成一个有效的训练序列，因此序列的采样非常简单。由于所有训练序列长度相同，无需进行填充操作，这提高了硬件利用率（同时也可以增大批次大小 $B$）。最后，我们也不必一次性将整个数据集全部加载到内存中即可进行训练样本的抽取，从而更容易处理那些可能无法完全放入内存的大型数据集。

**问题（data_loading）：实现数据加载（2分）**
交付要求：编写一个函数，该函数接收一个 numpy 数组 $x$（包含 token ID 的整数数组）、batch_size、context_length 和一个 PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'），并返回一对张量：采样的输入序列及其对应的下一个 token 目标。这两个张量的形状都应为 (batch_size, context_length)，包含 token ID，并且都应放置在指定的设备上。

低资源/降级训练提示：在 CPU 上进行数据加载
如果你计划在 CPU 上训练你的语言模型，你需要将数据移动到正确的设备上（同样地，之后你的模型也应使用相同的设备）。如果使用 CPU，可使用设备字符串 'cpu'。

代码可见[dataloader.py](hw5/dataloader.py)

如果数据集太大而无法全部加载到内存中怎么办？我们可以使用一个名为 mmap 的 Unix 系统调用，它能将磁盘上的文件映射到虚拟内存，并在访问对应内存位置时才加载文件内容。因此，你可以“假装”整个数据集已经在内存中了。NumPy 通过 np.memmap 提供了这一功能（或者在使用 np.load 时设置参数 mmap_mode='r'，前提是你之前是用 np.save 保存的数组），这会返回一个类数组对象，只有在你访问具体元素时才会按需加载数据。

在训练过程中从数据集（即 NumPy 数组）采样时，务必以内存映射模式加载数据集（通过 np.memmap 或 np.load 的 mmap_mode='r' 参数，具体取决于你保存数组的方式）。同时，请确保指定的 dtype 与你所加载的数组的原始数据类型一致。为确保安全，建议显式验证内存映射的数据是否正确（例如，检查数据中是否包含超出预期词表大小的非法 token ID 值）

### 5.2 检查点（Checkpointing）
除了加载数据外，我们还需要在训练过程中保存模型。在运行训练任务时，我们常常希望能够在训练中途意外停止后（例如由于任务超时、机器故障等）恢复训练。即使一切顺利，我们也可能希望之后能够访问训练过程中的中间模型（例如，事后研究训练动态、从不同训练阶段的模型中生成样本等）。

一个检查点（checkpoint）应包含所有能够恢复训练所需的状态。最基本的是，我们必须能够恢复模型的权重。如果使用了带有状态的优化器（例如 AdamW），我们还需要保存优化器的状态（例如，AdamW 中的动量估计值）。最后，为了能够恢复学习率调度，我们还需要知道训练停止时所处的迭代次数。PyTorch 使得保存这些信息变得非常简单：每个 `nn.Module` 都有一个 `state_dict()` 方法，它会返回一个包含所有可学习参数的字典；之后我们可以通过对应的 `load_state_dict()` 方法恢复这些权重。优化器 `torch.optim.Optimizer` 也同样支持 `state_dict()` 和 `load_state_dict()` 方法。最后，`torch.save(obj, dest)` 可以将一个对象（例如，字典，其值中包含张量，也可以包含整数等普通 Python 对象）序列化并保存到文件（路径）或类文件对象中，之后可以通过 `torch.load(src)` 将其重新加载回内存。

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

代码可见[checkpoint.py](hw5/checkpoint.py)

### 5.3 训练循环
现在，是时候将你之前实现的所有组件整合到你的主训练脚本中了。为了方便后续多次运行训练（以研究不同参数选择对训练的影响），建议将训练过程配置为可通过命令行参数轻松启动，并支持不同的超参数设置。

**问题（training_together）：整合所有组件（4分）**
交付要求： 编写一个脚本，运行训练循环，使用用户提供的输入来训练你的模型。具体来说，我们建议你的训练脚本至少具备以下功能：
- 可配置和控制各种模型及优化器的超参数。
- 使用 np.memmap 以内存高效的方式加载大型训练和验证数据集。
- 将检查点序列化保存到用户指定的路径。
- 定期记录训练和验证性能（例如，输出到控制台和/或外部服务如 Weights and Biases）。

代码可见[training_loop.py](hw5/training_loop.py)

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

**问题（decoding）：解码（3分）**
交付内容：实现一个从你的语言模型中进行解码的函数。我们建议你支持以下功能：
- 为用户提供提示词生成补全内容（即输入一段文本 $x_{1\ldots t}$，然后采样生成后续内容，直到生成结束标记 ）。
- 允许用户控制生成的最大 token 数量。
- 给定一个指定的温度值，对预测的下一个词分布应用 softmax 温度缩放后再进行采样。
- 支持 Top-p 采样（Holtzman 等，2020；也称为核采样），给定用户指定的阈值。

代码可见[inference.py](hw6/inference.py)

## 7 Experiments
现在是时候将所有内容整合起来，并在预训练数据集上训练（小型）语言模型了。

### 7.1 如何运行实验并提交成果
要真正理解 Transformer 架构组件背后的设计原理，最好的方法就是亲自修改并运行它。动手实践是无可替代的。

为此，能够快速、一致地进行实验，并记录下所做的一切至关重要。为了实现快速实验，我们将使用一个小型模型（1700万参数）和一个简单的数据集（TinyStories）进行多次小规模实验。为了确保实验的一致性，你需要以系统化的方式对各个组件进行消融分析，并调整超参数。同时，为了保留实验记录，我们要求你提交一份实验日志，以及每次实验对应的学习曲线。

为了能够提交损失曲线，请务必定期评估验证集上的损失，并记录训练的步数和实际耗时（wallclock time）。你可能会发现像 Weights and Biases 这样的日志记录工具非常有帮助。

**问题（experiment_log）：实验日志记录（3分）**
为你的训练和评估代码建立实验跟踪基础设施，以便能够根据梯度更新步数（gradient steps）和实际耗时（wallclock time）来追踪实验过程和损失曲线。
交付内容：实验的日志记录基础设施代码，以及一份实验日志（即本节后续作业问题中你所尝试的所有内容的记录文档）。

### 7.2 TinyStories
我们将从一个非常简单的数据集（TinyStories；Eldan 和 Li，2023）开始，在该数据集上模型训练速度很快，同时我们也能观察到一些有趣的行为。获取该数据集的说明请参见第1节。

**超参数调优**
我们会为你提供一些基本的超参数作为起点，然后请你通过实验找出其他超参数的良好设置。
- **vocab_size（词汇表大小）**: 10000。典型的词汇表大小通常在几万到几十万之间。你可以尝试调整该值，观察词汇表大小对模型行为的影响。
- **context_length（上下文长度）**: 256。像 TinyStories 这样的简单数据集可能不需要很长的序列长度，但在后续使用 OpenWebText 数据时，你可能需要调整该值。尝试改变上下文长度，观察其对每次迭代的运行时间和最终困惑度（perplexity）的影响。
- **d_model（模型维度）**: 512。这比许多小型 Transformer 论文中常用的 768 维略小，但可以加快训练速度。
- **d_ff（前馈网络维度）**: 1344。这大约是 d_model 的 8/3 倍，同时是 64 的倍数，有利于 GPU 性能优化。
- **RoPE theta 参数（Θ）**: 10000。
- **层数与注意力头数**: 4 层，16 个注意力头。这样的配置总共约有 1700 万个非嵌入参数，是一个相对较小的 Transformer 模型。
- **总处理 token 数**: 327,680,000（你的 batch size × 总训练步数 × 上下文长度 应大致等于该值）。

你需要通过一些试错，为以下超参数找到合适的默认值：
- 学习率（learning rate）
- 学习率预热步数（learning rate warmup）
- 其他 AdamW 超参数（$\beta_1$, $\beta_2$, $\epsilon$）
- 权重衰减（weight decay）

这些超参数的典型取值可参考 Kingma 和 Ba [2015] 的论文。

**整合所有内容**
现在，你可以将所有部分整合起来：获取一个训练好的 BPE 分词器，对训练数据集进行分词，并在你编写的训练循环中运行模型。
重要提示：如果你的实现正确且高效，上述超参数在 1 块 H100 GPU 上的运行时间应大约为 30–40 分钟。如果你的运行时间显著更长，请检查你的数据加载、模型保存或验证损失计算代码是否存在性能瓶颈，并确保你的实现已正确地进行了批处理（batched）。

代码可见[final_train.py](hw7/final_train.py)

**文本生成**
现在你已经拥有了训练好的解码器，我们可以开始生成文本了！我们将基于模型进行文本生成，并评估其生成质量。作为参考，你生成的文本至少应达到如下示例的水平：
示例（ts_generate_example）：TinyStories 语言模型的生成样本
![](../figures/fig8.png)
**低资源/降配提示：在 CPU 或 Apple Silicon 上生成文本**
如果你使用的是低资源配置（仅处理了 4000 万个 token），你生成的文本应仍能大致符合英语语法结构，但流畅度和连贯性不如上述高资源训练的模型。例如，我们在 40M token 配置下训练的 TinyStories 语言模型生成的样本如下：
![](../figures/fig9.png)

**问题（generate）：生成文本（1分）**
使用你的解码器和训练好的检查点，报告你的模型生成的文本。你可能需要调整解码器参数（如温度、top-p 等）以获得流畅的输出。
交付内容：至少 256 个 token 的文本输出（或直到第一个 token 为止），以及一段简要评论，说明该输出的流畅程度，以及至少两个影响此输出质量好坏的因素。

代码可见[final_inference.py](hw7/final_inference.py)

### 7.3 消融实验与架构修改
理解 Transformer 的最佳方式就是实际修改它，并观察其行为变化。现在我们将进行一些简单的消融实验和架构修改。

**消融实验1：层归一化（layer normalization）**
人们常说，层归一化对 Transformer 训练的稳定性至关重要。但也许我们想“铤而走险”。现在，让我们从每个 Transformer 模块中移除 RMSNorm，然后观察会发生什么。

**问题（layer_norm_ablation）：移除 RMSNorm 并训练（1分）（1 H100 小时）**
从你的 Transformer 中移除所有 RMSNorm 层并进行训练。在之前的最优学习率下会发生什么？你能否通过使用更低的学习率来获得稳定性？
交付内容：移除 RMSNorm 后进行训练的学习曲线，以及使用最佳学习率时的学习曲线。
交付内容：几句关于 RMSNorm 影响的简要评论。

---

现在让我们研究另一种在第一眼看来似乎随意的层归一化选择。Pre-norm Transformer块定义为：
$$
z = x + \mathrm{MultiHeadedSelfAttention}(\mathrm{RMSNorm}(x))
$$$$
y = z + \mathrm{FFN}(\mathrm{RMSNorm}(z)).
$$

这是对原始Transformer架构少数几个“共识”修改之一，原始架构采用的是post-norm方法，其形式如下：
$$
z = \mathrm{RMSNorm}\!\left(x + \mathrm{MultiHeadedSelfAttention}(x)\right)
$$$$
y = \mathrm{RMSNorm}\!\left(z + \mathrm{FFN}(z)\right).
$$

让我们回到post-norm方法，看看会发生什么。

**问题（pre_norm_ablation）：实现并训练后归一化模型（1分）（1个H100小时）**
将你的预归一化Transformer实现修改为后归一化。使用后归一化模型进行训练，观察会发生什么。
交付成果：后归一化Transformer的学习曲线，并与预归一化模型进行比较。

我们看到，层归一化对Transformer的行为有重大影响，甚至层归一化的位置也非常重要。

---

**消融实验2：位置嵌入**
接下来，我们将研究位置嵌入对模型性能的影响。具体来说，我们将把基础模型（使用RoPE）与完全不包含位置嵌入（NoPE）的情况进行比较。事实证明，仅解码器的Transformer（即我们所实现的带有因果掩码的模型）理论上可以在没有显式提供位置嵌入的情况下推断出相对或绝对的位置信息 [Tsai et al., 2019, Kazemnejad et al., 2023]。现在我们将通过实验验证NoPE与RoPE相比的表现如何。

**问题（no_pos_emb）：实现无位置嵌入（NoPE）（1分）（1个H100小时）**
将你使用RoPE的Transformer实现修改为完全移除位置嵌入信息，观察会发生什么。
交付成果：一条对比RoPE与NoPE性能的学习曲线。

---

**消融实验3：SwiGLU 与 SiLU**
接下来，我们将遵循 Shazeer [2020] 的方法，通过比较使用 SwiGLU 的前馈网络与仅使用 SiLU 激活函数但不包含门控线性单元（GLU）的前馈网络的性能，来测试前馈网络中门控机制的重要性：
$$
\mathrm{FFN}_\mathrm{SLU}(x)=W_2\mathrm{SiLU}(W_1x)
$$

回顾一下，在我们的 SwiGLU 实现中，我们将内部前馈层的维度设置为大约 $d_{ff}=\frac{8}{3}d_{model}$（同时确保 $d_{ff}~mod~64=0$，以便利用 GPU 的张量核心）。在你的 $\mathrm{FFN}_\mathrm{SLU}$ 实现中，应将 $d_{ff}=4 \times d_{model}$，以大致匹配 SwiGLU 前馈网络的参数数量（SwiGLU 使用三个权重矩阵，而此处为两个）。

**问题（swiglu_ablation）：SwiGLU 与 SiLU 对比（1分）（1个H100小时）**
交付成果：一条对比 SwiGLU 与 SiLU 前馈网络性能的学习曲线，两者参数量大致相当。
交付成果：几句话，简要讨论你的发现。
低资源/降规模提示：在线且GPU资源有限的学生应在TinyStories上测试修改

在本作业的后续部分，我们将转向更大规模、噪声更多的网络数据集（OpenWebText），实验不同的架构修改，并（可选）向课程排行榜提交结果。
在OpenWebText上将语言模型训练到流利程度需要很长时间，因此我们建议GPU资源有限的在线学生继续在TinyStories上测试修改（使用验证损失作为评估性能的指标）。

### 7.4 OpenWebText
我们现在将转向一个更为标准的、由网络爬取数据构建的预训练数据集。OpenWebText [Gokaslan et al., 2019] 的一个小样本也以单个文本文件的形式提供：参见第1节了解如何访问该文件。
注意：你可能需要为此实验重新调整你的超参数，例如学习率或批量大小。

**问题（main_experiment）：在OWT上的实验（2分）（3个H100小时）**
使用与TinyStories相同的模型架构和总训练步数，在OpenWebText上训练你的语言模型。该模型表现如何？

交付成果：你的语言模型在OpenWebText上的学习曲线。描述与TinyStories相比损失值的差异——我们应如何解释这些损失？

交付成果：从OpenWebText语言模型生成的文本，格式与TinyStories的输出相同。这段生成文本的流畅性如何？尽管我们使用了与TinyStories相同的模型和计算预算，为什么输出质量更差？