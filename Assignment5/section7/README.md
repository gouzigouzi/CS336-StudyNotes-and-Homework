# CS336 Assignment 5 (alignment): Alignment and Reasoning RL

## 7 Group Relative Policy Optimization
下文将介绍“组相对策略优化（GRPO）”——这是一种策略梯度变体，你将基于该方法实现并实验数学问题求解。

### 7.1 GRPO 算法
**优势值估计（Advantage Estimation）**
GRPO 的核心思想是：对每个问题，从策略 $\pi_{\theta}$ 中采样多个输出，并用这些输出计算基线。这种方式的优势在于：无需学习神经网络价值函数 $V_{\phi}(s)$ （该函数训练难度大，且从系统实现角度看较为繁琐）。传统方法里，通常需要训练一个额外的全尺寸神经网络（如 Critic/Value 模型）来专门预测基线，这非常耗时耗力。

为了省去专门训练一个预测网络，GRPO 采用了“组内对比”的方法来计算基线和优势值，通过让模型对同一问题回答多次并计算均值，巧妙地获得了一个评判标准（基线），从而计算出每一次回答的相对好坏（优势值），以此来更新模型。面对同一个数学题，它让当前模型一口气生成 $G$ 个答案，分别算出它们的真实得分 $r^{(1)}, r^{(2)}, \dots, r^{(G)}$。对于问题 $q$ 和从策略中采样的 $G$ 个输出 $\{o^{(i)}\}_{i=1}^G \sim \pi_{\theta}(\cdot | q)$，设 $r^{(i)} = R\left(q, o^{(i)}\right)$ 为第 $i$ 个输出的奖励。DeepSeekMath [Shao 等, 2024] 和 DeepSeek R1 [DeepSeek-AI 等, 2025] 中，第 $i$ 个输出的“组归一化奖励”（即优势值）定义为：
$$A^{(i)} = \frac{r^{(i)} - \text{mean}\left(r^{(1)}, r^{(2)}, \dots, r^{(G)}\right)}{\text{std}\left(r^{(1)}, r^{(2)}, \dots, r^{(G)}\right) + \text{advantage\_eps}}, \tag{28}$$

分子$r^{(i)} - \text{mean}\left(r^{(1)}, r^{(2)}, \dots, r^{(G)}\right)$ 计算第 $i$ 个答案比平均分高了多少（或低了多少），这就是最基础的优势值。如果是正数，接下来就会鼓励模型多生成这种答案；如果是负数，就会惩罚。分母 $\text{std}(\dots)$表示这 $G$ 个答案得分的标准差。除以标准差是为了做数据归一化（防止得分差距过大导致梯度爆炸），让优势值稳定在一个合理的区间内。advantage_eps 为防止分母为 0 的小常数。注意：该优势值 $A^{(i)}$ 对响应中的每个 token 都相同，即$A_t^{(i)} = A^{(i)}, \forall t \in \{1, ..., |o^{(i)}|\}$，因此下文将省略下标 $t$。在数学解题这类任务中，模型通常是在生成完所有的中间推理和最终答案后，才会获得一次整体的奖励（对或错）。因为系统无法自动将这个最终奖励精确分配给中间的某一个特定词，所以算法选择将评估整个回答算出的总优势值 $A^{(i)}$，无差别地赋予该回答中的每一步（每一个 token）。

**高层算法流程（High-level algorithm）**
在深入 GRPO 目标函数之前，先通过 Shao 等 [2024] 提出的算法 3，了解 GRPO 的训练循环框架。注 ：这是 DeepSeekMath 中 GRPO 的特例——使用经过验证的奖励函数，无 KL 散度项，也没有对参考模型和奖励模型进行迭代更新。

**GRPO 目标函数 （GRPO objective）**
GRPO 目标函数融合了三项核心思想：
1. 异策略策略梯度（见式 27）；
2. 通过组归一化计算优势值$A^{(i)}$（见式 28）；
3. 裁剪机制（Clipping Mechanism），源自近邻策略优化（PPO, Schulman 等 [2017]）。

裁剪机制的目的是：在同一批轨迹上执行多步梯度更新时，保证训练稳定性。它的工作原理是防止当前策略 $\pi_{\theta}$ 偏离旧策略太远。

![](../figures/fig4.png)

首先，我们给出完整的 GRPO-Clip 目标函数，再解释裁剪操作的作用：
![](../figures/fig5.png)

超参数 $\epsilon>0$ 用于控制策略的更新幅度。为更直观理解，我们参考 Achiam [2018a,b]的方法重写逐 token 目标函数。定义函数：
$$
g\left(\epsilon, A^{(i)}\right)=
\begin{cases}
(1+\epsilon)A^{(i)}, & \text{if } A^{(i)}\ge 0,\\
(1-\epsilon)A^{(i)}, & \text{if } A^{(i)}< 0.
\end{cases} \tag{30}
$$

则逐 token 目标函数可重写为：
$$
\text{per-token objective}=
\min\!\left(
\frac{\pi_{\theta}\!\big(o^{(i)}_{t}\mid q, o^{(i)}_{<t}\big)}
     {\pi_{\theta_{\mathrm{old}}}\!\big(o^{(i)}_{t}\mid q, o^{(i)}_{<t}\big)}A^{(i)},
\;
g\left(\epsilon, A^{(i)}\right)
\right).
$$

我们分情况讨论：
- 当优势值 $A(i)$ 为正时，逐 token 目标函数简化为：
$$
\text{per-token objective}=\min\!\left(
\frac{\pi_{\theta}\!\big(o^{(i)}_{t}\mid q, o^{(i)}_{<t}\big)}
     {\pi_{\theta_{\mathrm{old}}}\!\big(o^{(i)}_{t}\mid q, o^{(i)}_{<t}\big)},
\,1+\epsilon
\right)A^{(i)}.
$$
由于 $A^{(i)}>0$，若动作 $o_{t}^{(i)}$ 在 $\pi_{\theta}$ 下的概率增大（即 ${\pi_{\theta}\left(o_{t}^{(i)} | q, o_{<t}^{(i)}\right)}$)的值变大），目标函数值会增加。min 函数的裁剪作用限制了目标函数的增长幅度：当 ${\pi_{\theta}\left(o_{t}^{(i)} | q, o_{<t}^{(i)}\right)} > (1+\epsilon){\pi_{\theta_{old}}\left(o_{t}^{(i)} | q, o_{<t}^{(i)}\right)}$) 时，逐 token 目标函数达到最大值 $(1+\epsilon)A(i)$，从而避免策略 $\pi_\theta$ 与旧策略 $\pi_{\theta_{old}}$ 偏差过大。

- 当优势值 $A(i)$ 为负时，模型会尝试降低 ${\pi_{\theta}\left(o_{t}^{(i)} | q, o_{<t}^{(i)}\right)}$ 的概率，但裁剪机制会阻止其降至 $(1-\epsilon){\pi_{\theta_{old }}\left(o_{t}^{(i)} | q, o_{<t}^{(i)}\right)}$ 以下（完整推导参见Achiam [2018b]）。

### 7.2 Implementation
在理解 GRPO 的训练流程和目标函数后，我们开始分模块实现。SFT 和 EI 部分的许多模块可直接复用。

**计算优势值（组归一化奖励）Computing advantages (group-normalized rewards)**
首先实现 a rollout batch 中每个样本的优势值计算逻辑，即组归一化奖励。我们考虑两种组归一化方式：1. 前文公式 28 的标准方法. 2. 近期提出的简化方法。
Dr. GRPO [Liu et al., 2025] 指出，通过 $std(r(1), r(2), \dots, r(G))$ 进行归一化的方式，会奖励答案正确性波动较小的问题，而这可能并不理想。因此，他们提出移除标准差归一化步骤，直接计算：
$$A^{(i)} = r^{(i)} - \text{mean}\left(r^{(1)}, r^{(2)}, ..., r^{(G)}\right) \tag{31}$$

我们将实现两种变体，并在后续实验中对比其性能。

**问题（compute_group_normalized_rewards）：组归一化（2分）**
交付要求：实现 `compute_group_normalized_rewards` 方法，计算每个滚动响应的原始奖励，在组内进行归一化，并返回归一化奖励、原始奖励及有用的元数据。
推荐接口：
```python
def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """为每组 rollout 响应计算奖励，并按组进行归一化。

    参数：
    reward_fn: Callable[[str, str], dict[str, float]]  
        用于将 rollout 响应与标准答案（ground truth）进行比较并打分的函数，返回一个字典，
        包含键 "reward"、"format_reward" 和 "answer_reward"。
    
    rollout_responses: list[str]  
        策略生成的 rollout 响应列表。该列表长度为 rollout_batch_size，
        即 rollout_batch_size = n_prompts_per_rollout_batch * group_size。
    
    repeated_ground_truths: list[str]  
        每个样本对应的标准答案列表。该列表长度也为 rollout_batch_size，
        因为每个问题的标准答案被重复了 group_size 次（与每个问题对应的多个响应对齐）。
    
    group_size: int  
        每个问题（即每组）生成的响应数量。
    
    advantage_eps: float  
        用于归一化时避免除零的小常数。
    
    normalize_by_std: bool  
        若为 True，则用每组奖励的标准差进行归一化（即减去均值后除以标准差）；
        否则仅减去组内均值。

    返回：
    tuple[torch.Tensor, torch.Tensor, dict[str, float]]
        - advantages: shape (rollout_batch_size,)，每条 rollout 响应的组内归一化奖励（即优势值）。
        - raw_rewards: shape (rollout_batch_size,)，每条 rollout 响应的原始未归一化奖励。
        - metadata: 用户自定义的其他统计信息，可用于日志记录（例如奖励的均值、标准差、最大/最小值等）。
    """
```
测试方法：实现 `[adapters.run_compute_group_normalized_rewards]`，运行命令 `uv run pytest -k test_compute_group_normalized_rewards` 并确保测试通过。

代码可见 [run_compute_group_normalized_rewards.py](run_compute_group_normalized_rewards.py)

**朴素策略梯度损失**
接下来实现损失计算相关方法。需注意：这些并非传统意义上的损失函数，不应作为评估指标。在强化学习中，应跟踪训练集和验证集的回报值等指标（详见6.5节讨论）。

首先实现朴素策略梯度损失，该损失直接将优势值与动作的对数概率相乘并取负。对于问题 q、响应 o 和响应 token $o_t$，逐 token 朴素策略梯度损失为：
$$-A_{t} \cdot \log p_{\theta}\left(o_{t} | q, o_{<t}\right) \tag{32}$$

**问题（compute_naive_policy_gradient_loss）：朴素策略梯度（1分）**
交付要求：实现 `compute_naive_policy_gradient_loss` 方法，使用原始奖励或预计算的优势值计算逐 token 策略梯度损失。
推荐接口：
```python
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    计算每个token的策略梯度损失，其中raw_rewards_or_advantages可为原始奖励或已归一化的优势值
    
    参数：
        raw_rewards_or_advantages: 形状为(batch_size, 1)的张量，每个滚动响应的标量奖励/优势值
        policy_log_probs: 形状为(batch_size, sequence_length)的张量，每个token的对数概率
    
    返回：
        形状为(batch_size, sequence_length)的张量，逐token策略梯度损失（将在训练循环中跨批次和序列维度聚合）
    """
```
实现提示：
- 将 raw_rewards_or_advantages 在 sequence_length 维度上广播（broadcast）

测试方法：实现 `[adapters.run_compute_naive_policy_gradient_loss]`，运行命令 `uv run pytest -k test_compute_naive_policy_gradient_loss` 并确保测试通过。

代码可见 [run_compute_naive_policy_gradient_loss.py](run_compute_naive_policy_gradient_loss.py)

**GRPO-Clip 损失**
接下来实现更核心的 GRPO-Clip 损失。逐 token GRPO-Clip 损失为：
$$-\min\left( \frac{\pi_{\theta}\left(o_{t} | q, o_{<t}\right)}{\pi_{\theta_{old}}\left(o_{t} | q, o_{<t}\right)} A_{t}, clip\left( \frac{\pi_{\theta}\left(o_{t} | q, o_{<t}\right)}{\pi_{\theta_{old}}\left(o_{t} | q, o_{<t}\right)},1-\epsilon,1+\epsilon \right) A_{t} \right)$$

**问题（compute_grpo_clip_loss）：GRPO-Clip损失（2分）**
交付要求：实现 `compute_grpo_clip_loss` 方法，计算逐 token GRPO-Clip 损失。

推荐接口：
```python
def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """    
    参数：
        advantages: 形状为(batch_size, 1)的张量，每个样本的优势值A
        policy_log_probs: 形状为(batch_size, sequence_length)的张量，待训练策略的逐token对数概率
        old_log_probs: 形状为(batch_size, sequence_length)的张量，旧策略的逐token对数概率
        cliprange: 裁剪参数ε（例如0.2）
    
    返回：
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: 形状为(batch_size, sequence_length)的张量，逐token裁剪损失
            metadata: 需记录的元数据（建议记录每个token是否被裁剪，即min函数右侧的裁剪后损失是否小于左侧）
    """
```
实现提示：
- 将优势值在sequence_length维度上广播（broadcast）

测试方法：实现 `[adapters.run_compute_grpo_clip_loss]`，运行命令 `uv run pytest -k test_compute_grpo_clip_loss` 并确保测试通过。

代码可见 [run_compute_grpo_clip_loss.py](run_compute_grpo_clip_loss.py)

**策略梯度损失包装器**
我们将通过对比实验验证三种策略梯度变体：
(a) 无基线（no_baseline）：无基线的朴素策略梯度损失，优势值直接为原始奖励 $A=R(q, o)$
(b) reinforce_with_baseline：使用组归一化奖励作为优势值（advantage）的朴素策略梯度损失。如果 $\bar{r}$ 是来自 compute_group_normalized_rewards 的组归一化奖励（可能已或未按组标准差归一化），那么优势值 $A = \bar{r}$。
（c）grpo_clip：GRPO-Clip 损失函数

为方便起见，我们将实现一个包装器（wrapper），使我们能够轻松在这三种策略梯度损失函数之间切换。

**问题（compute_policy_gradient_loss）：策略梯度包装器（1分）**
交付要求：实现 `compute_policy_gradient_loss` 函数——一个便捷包装器，用于调度至对应的损失计算流程（`no_baseline`、`reinforce_with_baseline` 或 `grpo_clip`），并返回逐 token 损失（per-token loss）及所有辅助统计信息。

推荐接口如下：
```python
def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    advantages: torch.Tensor | None = None,
    raw_rewards: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```
功能：选择并计算目标策略梯度损失。

参数说明：
- `policy_log_probs`：形状为（batch_size, sequence_length），表示待训练策略的逐 token 对数概率。
- `loss_type`：损失类型，可选值为 “no_baseline”、“reinforce_with_baseline” 或 “grpo_clip”。
- `raw_rewards`：当 loss_type == "no_baseline" 时为必填项，形状为（batch_size, 1）。
- `advantages`：当 loss_type 为 “reinforce_with_baseline” 或 “grpo_clip” 时为必填项，形状为（batch_size, 1）。
- `old_log_probs`：当 loss_type == "grpo_clip" 时为必填项，形状为（batch_size, sequence_length）。
- `cliprange`：当 loss_type == "grpo_clip" 时为必填项，用于裁剪的标量参数 ϵ。

返回值：
- 元组（torch.Tensor, dict[str, torch.Tensor]）：
   1. 损失张量（loss）：形状为（batch_size, sequence_length），即逐token损失。
   2. 元数据字典（metadata）：包含底层计算流程的统计信息（例如 GRPO-Clip 的裁剪比例）。

实现提示：
1. 调用 `compute_naive_policy_gradient_loss` 或 `compute_grpo_clip_loss` 完成具体计算。
2. 执行参数校验（参考上述断言模式）。
3. 将所有返回的元数据汇总到单个字典中。

测试方式：实现 `[adapters.run_compute_policy_gradient_loss]`，运行命令 `uv run pytest -k test_compute_policy_gradient_loss` 并验证测试通过。

代码可见 [run_compute_policy_gradient_loss.py](run_compute_policy_gradient_loss.py)

**掩码均值（Masked Mean）**
截至目前，我们已具备计算优势函数（advantages）、对数概率、逐 token 损失，以及逐 token 熵、裁剪比例等辅助统计信息的计算能力。为将形状为（batch_size, sequence_length）的 per-token loss tensors 缩减为损失向量（每个样本对应一个标量损失），我们将在序列维度上计算损失的均值，但仅包含响应对应的索引（即 mask[i, j]==1 的token位置）。

在大多数基于大语言模型（LLM）的强化学习（RL）代码库中，按序列长度归一化是标准操作，但这一做法的合理性尚未明确——观察公式（21）中策略梯度估计的定义可知，其中并不存在归一化因子 $\frac{1}{T^{(i)}}$。我们将先采用这一标准方法（通常称为 `masked_mean`），后续再测试 SFT 阶段实现的 `masked_normalize` 方法。

该函数支持**指定**均值计算的维度：若 `dim = None`，则对所有掩码为 1 的元素计算均值。这一功能可用于获取响应 token 的平均逐 token 熵、裁剪比例等统计信息。

**问题（masked_mean）：掩码均值（1分）**
交付要求：实现 `masked_mean` 方法，在尊重布尔掩码（boolean mask）的前提下对张量元素求平均。

推荐接口如下：
```python
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
```
功能：沿指定维度计算张量的均值，仅考虑 `mask == 1` 的元素。

参数说明：
- `tensor`：待求平均的输入数据张量。
- `mask`：与 tensor 形状相同的布尔掩码张量，值为1的位置将纳入均值计算。
- `dim`：计算均值的维度；若为 None，则对所有掩码为1的元素计算全局均值。

返回值：
- 掩码均值张量（torch.Tensor）：形状与 `tensor.mean(dim)` 的输出语义一致。

测试方式：实现 `[adapters.run_masked_mean]`，运行命令 `uv run pytest -k test_masked_mean` 并确保测试通过。

代码可见 [run_masked_mean.py](run_masked_mean.py)

**GRPO 微批次训练步骤（GRPO Microbatch Train Step）**
现在我们可以实现 GRPO 的单个 microbatch train step（回想一下，对于一个训练小批次，若 `gradient_accumulation_steps` > 1，我们会迭代多个 microbatch）。

具体而言，给定原始奖励（raw rewards）或优势函数（advantages）及对数概率，我们将计算逐 token 损失，通过 masked_mean 聚合为每个样本的标量损失，在批次维度上求平均，根据梯度累积步数调整损失，并执行反向传播。

**问题（grpo_microbatch_train_step）：微批次训练步骤（3分）**
交付要求：实现 GRPO 的单个微批次更新，包括策略梯度损失计算、掩码均值聚合及梯度缩放。

推荐接口如下：
```python
def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```
功能：对单个微批次执行前向传播与反向传播。

参数说明：
- `policy_log_probs`：形状为（batch_size, sequence_length），表示待训练策略的逐 token 对数概率。
- `response_mask`：形状为（batch_size, sequence_length），响应 token 位置标记为 1，提示词（prompt）/ 填充（padding）token 位置标记为 0。
- `gradient_accumulation_steps`：每个优化器步骤对应的微批次数量。
- `loss_type`：损失类型，可选值为 “no_baseline”、“reinforce_with_baseline” 或 “grpo_clip”。
- `raw_rewards`：当 loss_type == "no_baseline" 时为必填项，形状为（batch_size, 1）。
- `advantages`：当 loss_type != "no_baseline" 时为必填项，形状为（batch_size, 1）。
- `old_log_probs`：当 loss_type == "grpo_clip" 时为必填项，形状为（batch_size, sequence_length）。
- `cliprange`：GRPO-Clip 策略的裁剪参数 ϵ。

返回值：
- 元组（torch.Tensor, dict[str, torch.Tensor]）：
  1. 标量损失张量（loss）：经梯度累积调整后的微批次损失，用于日志记录。
  2. 元数据字典（metadata）：包含底层损失计算的元数据及其他需日志记录的统计信息。

实现提示：
1. 需在该函数中调用 `loss.backward()`，并确保根据梯度累积步数调整损失。

测试方式：实现 `[adapters.run_grpo_microbatch_train_step]`，运行命令 `uv run pytest -k test_grpo_microbatch_train_step` 并确认测试通过。

代码可见 [run_grpo_microbatch_train_step.py](run_grpo_microbatch_train_step.py)

**整合所有模块：GRPO 训练循环（Putting it all together: GRPO Train Loop）**
现在我们将整合所有模块，实现完整的 GRPO 训练循环。请参考 7.1 节的算法框架，合理调用已实现的方法。

以下提供初始超参数配置：若实现正确，使用该配置应能获得合理结果。
```python
n_grpo_steps: int = 200
learning_rate: float = 1e-5
advantage_eps: float = 1e-6
rollout_batch_size: int = 256  # 滚动/采样批次大小。这一轮采样总共生成了多少个回答。
group_size: int = 8  # 组大小。针对同一个提示词 (Prompt)，模型独立生成的回答数量。
sampling_temperature: float = 1.0
sampling_min_tokens: int = 4  # 参考 Expiter，禁止空字符串响应
sampling_max_tokens: int = 1024
epochs_per_rollout_batch: int = 1  # 在线策略（On-policy）
train_batch_size: int = 256  # 在线策略，训练总批次
gradient_accumulation_steps: int = 128  # 微批次大小为 2，可在 H100 显卡上运行，梯度累加步数。显卡一次吃不下 256 条，我们就把它拆成 128 份，分 128 次喂给显卡。每次只计算梯度，但不更新参数，把梯度在内存里累加起来。
gpu_memory_utilization: float = 0.85
loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
use_std_normalization: bool = True

optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=learning_rate,
    weight_decay=0.0,
    betas=(0.9, 0.95),
)
```
上述默认超参数适用于**在线策略（On-policy）** 场景：每个滚动批次（rollout batch）仅执行一次梯度更新。此时需满足 `train_batch_size == rollout_batch_size` 且 `epochs_per_rollout_batch == 1`。在线策略（On-policy）意味着模型刚才自己做出来的题（Rollout 生成的数据），必须立刻拿来训练更新，绝不用旧数据。

以下提供一些合理性校验断言及常量定义，可规避部分边界情况并提供实现指引：
```python
assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size 必须能被 gradient_accumulation_steps 整除"
micro_train_batch_size = train_batch_size // gradient_accumulation_steps  # 微批次，这是显卡真正单次前向/反向传播处理的数据量。

assert rollout_batch_size % group_size == 0, "rollout_batch_size 必须能被 group_size 整除"
n_prompts_per_rollout_batch = rollout_batch_size // group_size  # 提示词数量。这一轮一共抽了多少道不一样的题（Prompt）给模型做

assert train_batch_size >= group_size, "train_batch_size 必须大于或等于 group_size"

n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size  # 计算在一个完整的“采样生成批次”（rollout batch）的数据量里，一共包含了多少个用于显卡前向/反向传播计算的“微小训练批次”（micro-batch）。
```
以下是一些额外的建议：
- 记得使用 r1_zero prompt，并指示 vLLM 在遇到第二个 `</answer>` 标签时停止生成，如之前实验中所做的那样。
- 建议使用 typer 进行命令行参数解析。
- 使用梯度裁剪（gradient clipping），裁剪值设为 1.0。
- 应定期记录验证奖励（例如每 5 或 10 步记录一次）。在比较超参数时，应至少在 1024 个验证样本上进行评估，因为 CoT/RL 的评估结果可能存在较大噪声。
- 在我们当前的损失实现中，GRPO-Clip 仅应在离策略（off-policy）设置下使用（因为它需要旧的对数概率）。
- 在 off-policy 设置中，若对每个 rollout batch 执行多个 epoch 的梯度更新，每次都重新计算旧的对数概率是低效的。更高效的做法是只计算一次旧的对数概率，并在每个 epoch 中重复使用。
- 不应对旧的对数概率进行梯度求导。
- 在每次优化器更新时，应记录以下部分或全部指标：
  - 损失（loss）
  - 梯度范数（gradient norm）
  - token 熵（token entropy）
  - 裁剪比例（clip fraction），如果是离策略训练
  - 训练奖励（train rewards），包括总奖励、格式奖励和答案奖励
  - 任何你认为对调试有帮助的其他信息

**问题（grpo_train_loop）：GRPO训练循环（5分）**
交付要求：实现 GRPO 的完整训练循环。基于 MATH 数据集启动策略训练，确认验证奖励（validation rewards）逐步提升，且不同阶段的采样结果（rollouts）合理。提供验证奖励随训练步数变化的图表，并附上不同时期的若干采样示例。

代码可见 [run_grpo_train_loop.py](run_grpo_train_loop.py)