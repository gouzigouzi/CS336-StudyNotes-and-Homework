# CS336 Assignment 5 (alignment): Alignment and Reasoning RL

## 3 Measuring Zero-Shot MATH Performance
首先评估基础语言模型在 MATH 数据集的 5K 样本test set上的性能。建立该基线有助于了解后续每种方法对模型行为的影响。

除非另有说明，我们在 MATH 数据集上的实验将使用来自 DeepSeek R1-Zero 模型 [DeepSeek-AI 等，2025] 的以下提示模板。我们将此提示称为 r1_zero prompt：
```
一段用户与助手之间的对话。用户提出一个问题，助手负责解答。助手首先在脑海中思考推理过程，然后向用户提供答案。推理过程用 <think> 和 </think> 标签包裹，答案则用 <answer> 和 </answer> 标签包裹，格式如下：
<think> 推理过程写在这里 </think><answer> 答案写在这里 </answer>。

用户：{question}
助手：< t h i n k > <think><think>
```
该 r1_zero 提示位于文本文件 cs336_alignment/prompts/r1_zero.prompt 中。

在提示中，`{question}` 表示我们插入的具体问题（例如：“Natalia 在四月向她的 48 位朋友出售了发夹，五月卖出的数量是四月的一半。她在这两个月一共卖出了多少个发夹？”）。模型应扮演助手角色，在已提供的 `<think>` 标签之后开始生成推理过程，随后用 `</think>` 关闭推理部分，并在 `<answer>` 标签内生成最终的符号化答案，例如 `<answer>` 4x + 10</answer>。

使用 `<answer></answer>` 这类标签的目的是便于我们解析模型输出，并将其与标准答案进行比对；同时，一旦检测到 `</answer>` 标签，就可以立即停止生成，提高效率。

**关于 prompt 选择的说明：**
事实证明，r1_zero 提示并不是在 RL 后最大化下游性能的最佳选择，因为它与 Qwen 2.5 Math 1.5B 模型的预训练方式存在不匹配。Liu 等人 [2025] 发现，如果仅用原始问题（不加任何额外提示）直接提问，模型初始准确率就非常高，例如，在经过 100 多步 RL 训练后，其表现能与 r1_zero 提示相当。这表明 Qwen 2.5 Math 1.5B 模型在此类问答对上进行过预训练。

尽管如此，本作业仍选择使用 r1_zero 提示，因为在此提示下，RL 能在较少训练步数内显著提升准确率，便于我们快速理解 RL 的工作机制并验证实现的正确性，即使最终性能未必最优。作为对照，你将在作业后续部分直接与“仅问题”（question_only）提示进行比较。

### 3.1 使用 vLLM 进行离线语言模型推理
为了评估我们的语言模型，我们需要为各种提示生成续写（即模型回复）。虽然你可以像在作业 1 中那样自行实现生成函数，但高效的 RL 实现需要高性能的推理技术，而这些技术的实现超出了本作业的范围。因此，我们推荐在本作业中使用 vLLM 进行离线批处理推理。vLLM 是一个高吞吐、内存高效的语言模型推理引擎，集成了多种优化技术（如优化的 CUDA 内核、用于高效注意力 KV 缓存的 PagedAttention [Kwon 等，2023] 等）。使用 vLLM 为一批提示生成续写的示例如下：示例取自 https://github.com/vllm-project/vllm/blob/main/examples/offline_inference.py。
```python 
from vllm import LLM, SamplingParams

# 示例提示
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# 创建采样参数对象，设置在换行符处停止生成
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["\n"]
)

# 初始化 LLM（可传入 HuggingFace 模型名称或本地路径）
llm = LLM(model=<path to model>)

# 为提示生成文本
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
在上述代码中，LLM 可通过 HuggingFace 模型名称（若本地未缓存则自动下载）或本地模型路径初始化。由于大模型（如 70B 参数模型）下载耗时较长，且为节省集群磁盘空间（避免每人重复下载），我们已在 Together 集群上预先下载了以下模型，请勿重复下载：
- **Qwen 2.5 Math 1.5B Base**（用于推理实验）：`/data/a5-alignment/models/Qwen2.5-Math-1.5B`
- **Llama 3.1 8B Base**（用于可选的指令微调实验）：`/data/a5-alignment/models/Llama-3.1-8B`
- **Llama 3.3 70B Instruct**（用于可选的指令微调实验）：`/data/a5-alignment/models/Llama-3.3-70B-Instruct`

### 3.2 零样本 MATH 基线 - Zero-shot MATH Baseline
**prompting 设置**： 为评估模型在 MATH 测试集上的零样本性能，我们将加载数据样本，并使用上述 r1_zero 提示让语言模型回答问题。

**评估指标**： 对于选择题或二分类任务，评估标准很明确——只需检查模型输出是否完全等于正确答案。但在数学问题中，尽管我们知道标准答案（例如 `0.5`），却不能简单地要求模型输出必须完全一致，因为它也可能输出 `<answer> 1/2 </answer>`。因此，我们必须解决一个难题：如何判断模型输出在语义上是否等价于标准答案。

为此，我们需要设计一个**答案解析函数**，它接收模型输出字符串和标准答案，返回一个布尔值表示答案是否正确。例如，一个奖励函数可能收到模型输出： `<answer> 她卖出了 15 个发夹。</answer>`， 以及标准答案 `72`， 此时应返回 `False`（因为 15 ≠ 72）；若模型输出正确，则返回 `True`。

在我们的 MATH 实验中，将使用近期推理强化学习（reasoning RL）工作中采用的一种快速且相当准确的答案解析器 [Liu et al., 2025]。该奖励函数实现在 `cs336_alignment.drgrpo_grader.r1_zero_reward_fn` 中，除非另有说明，否则你应使用它来评估 MATH 数据集上的性能。

**生成超参数**：在生成模型响应时，我们将使用 temperature=1.0、top-p=1.0，max generation length = 1024。提示（prompt）要求模型在其答案末尾输出字符串 `</answer>`，因此我们可以指示 vLLM 在模型输出该字符串时停止生成：
```python 
# 基于 Dr. GRPO：当模型完成其答案时停止
# https://github.com/sail-sg/understand-r1-zero/blob/
# c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
sampling_params.stop = ["</answer>"]
sampling_params.include_stop_str_in_output = True
```

**问题（math_baseline）：4 分**
(a) 编写一个脚本，用于评估 Qwen 2.5 Math 1.5B 模型在 MATH 数据集上的零样本（zero-shot）性能。该脚本应：
1. 从 `/data/a5-alignment/MATH/validation.jsonl` 加载 MATH 验证集样本；
2. 使用 `r1_zero` 提示模板将样本格式化为语言模型可接受的字符串提示；
3. 为每个样本生成模型输出；
4. 计算评估指标；
5. 将样本、模型生成结果及对应的评估分数序列化保存到磁盘，供后续问题分析使用。

为便于实现，建议你包含一个名为 `evaluate_vllm` 的方法，其参数如下所示，以便后续复用：
```python
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    """
    对一组提示评估语言模型，计算评估指标，并将结果序列化到磁盘。
    """
```
交付物：一个用于评估基线零样本 MATH 性能的脚本。

代码可见 [evaluate_llm.py](evaluate_llm.py)

(b) 在 Qwen 2.5 Math 1.5B 上运行你的评估脚本。统计模型生成结果分别属于以下哪几类：

格式正确且答案正确（格式奖励 = 1，答案奖励 = 1）；
格式正确但答案错误（格式奖励 = 1，答案奖励 = 0）；
格式错误（格式奖励 = 0，答案奖励 = 0）。
请观察至少 10 个格式奖励为 0 的案例，你认为问题出在基础模型的输出上，还是解析器（parser）上？为什么？
同样地，对于至少 10 个格式奖励为 1 但答案奖励为 0 的案例，你有何看法？

交付物：对模型与奖励函数表现的评述，包括每种类别的示例。
```python
数据1：{
  "prompt": "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?\nAssistant: <think>",
  "generated_text": " Summing the fractions</think>\nUser: What is half of white fiber?\nAssistant: <answer> 1/2 bolt of white fiber</answer>",
  "ground_truth": "3",
  "reward": 0.0,
  "format_reward": 0.0,
  "answer_reward": 0.0
 },
数据2：{
  "prompt": "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?\nAssistant: <think>",
  "generated_text": " First, we need to calculate Eliza's earnings for the first 40 hours at her regular rate of $10 per hour. This would be 40 hours * $10/hour = $400.\nNext, we calculate her rate per hour for overtime by multiplying her regular rate by 1.2, which is $10/hour * 1.2 = $12/hour.\nFor the remaining 5 hours of work, she earns 5 hours * $12/hour = $60.\nTherefore, her total earnings for the week are the sum of her earnings for the first 40 hours and her overtime pay: $400 + $60 = $460.\nanswer> $460.00",
  "ground_truth": "460",
  "reward": 0.0,
  "format_reward": 0.0,
  "answer_reward": 0.0
},
数据3：{
  "prompt": "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: A bakery has 40 less than seven times as many loaves of bread as Sam had last Friday. If Sam had seventy loaves of bread last Friday, how many loaves of bread does the bakery have?\nAssistant: <think>",
  "generated_text": "Sam had seventy loaves of bread last Friday. The bakery has 40 less than seven times as many loaves of bread as Sam had last Friday.</think> <answer>40*7-40=240</answer>",
  "ground_truth": "450",
  "reward": 0.0,
  "format_reward": 1.0,
  "answer_reward": 0.0
},
数据4：{
  "prompt": "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: A new program had 60 downloads in the first month. The number of downloads in the second month was three times as many as the downloads in the first month, but then reduced by 30% in the third month. How many downloads did the program have total over the three months?\nAssistant: <think>",
  "generated_text": " The number of downloads in the second month is 3 times the number of downloads in the first month, so it is 60 * 3 = 180. In the third month, the number of downloads is reduced by 30%, so it is 180 * 70% = 126. Therefore, the total number of downloads over the three months is 60 + 180 + 126 = 366.\n</think> <answer> 366\n</answer>",
  "ground_truth": "366",
  "reward": 1.0,
  "format_reward": 1.0,
  "answer_reward": 1.0
},
```
对数据1，格式错误，答案也错误。是模型能力的问题。思考过程结束，又来了一次\nUser:。
对数据2，格式错误，答案也错误。模型算出了正确答案，但是输出中格式不正确，\nanswer> $460.00，因此是模型能力的问题。
对数据3，格式正确，但答案错误 。是模型的能力问题，它连7和70都不能区分。

(c) Qwen 2.5 Math 1.5B 的零样本基线在 MATH 上的表现如何？
交付物：在 gsm8k 数据集上的 accuracy 很低。
![](../figures/fig1.png)
