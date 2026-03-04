# CS336 Assignment 5 (alignment): Alignment and Reasoning RL

## 8 GRPO实验
接下来我们将基于 GRPO 训练循环开展实验，尝试不同超参数和算法调整。每个实验需占用 2 块 GPU，一块用于 vLLM 实例，另一块用于策略训练。

**关于提前终止运行的说明**：若在 200 步 GRPO 训练前就发现超参数间存在显著差异（例如某一配置出现发散或明显次优），可提前终止该实验，为后续运行节省时间和计算资源。下文提及的GPU时长为粗略估计。

**问题（grpo_learning_rate）：学习率调优（2分）（6 个 H100 GPU 小时）**
以上述建议超参数为基准，对学习率进行扫描式调优（sweep），报告最终的验证答案奖励（若优化器发散则注明）。
交付要求：多个学习率对应的验证奖励曲线；在 MATH 数据集上验证准确率至少达到 25% 的模型；用 2 句话简要说明在其他记录指标中观察到的趋势。后续所有实验均采用本次调优中表现最佳的学习率。

使用 "reinforce_with_baseline" 模式，和 gsm8k 数据集中所有的原始训练数据，lr = [1e-5, 1e-3, 1e-4, 5e-5, 3e-5, 2e-5]。其中1e-3, 1e-4发散。最终选择 2e-5 作为最佳学习率。

**基线（baselines）的影响**
沿用上述超参数（学习率替换为最优值），研究基线的影响。本实验为在策略（on-policy）设置，需对比以下两种损失类型：
- 无基线（no_baseline）
- 带基线的强化学习（reinforce_with_baseline）

注：默认超参数中 use_std_normalization（标准差归一化）设为 True。

**问题（grpo_baselines）：基线的影响（2分）（2个H100 GPU小时）**
分别采用 reinforce_with_baseline 和 no_baseline 训练策略。
交付要求：每种损失类型对应的验证奖励曲线；用2句话简要说明在其他记录指标中观察到的趋势。

后续几个实验均采用本次实验中表现最佳的损失类型。
经过测试，有baseline的算法更好。

**长度归一化（Length normalization）**
如实现 masked_mean 函数时所提示，对序列长度进行损失平均并非必要，甚至可能不合理。损失的求和方式是一项重要超参数，其选择会影响策略动作的信用分配（credit attribution）方式。

下面结合 Lambert（2024）的示例进行说明。观察 GRPO 训练步骤，首先得到逐令牌（per-token）的策略梯度损失（暂不考虑裁剪）：
```python
advantages  # (批量大小, 1)，优势函数值
per_token_probability_ratios  # (批量大小, 序列长度)，逐令牌概率比
per_token_loss = -advantages * per_token_probability_ratios  # 逐令牌损失
```
其中优势函数值已沿序列长度维度广播（broadcasted）。以下对比两种逐令牌损失的聚合方式：
- 我们实现的 masked_mean：对每个序列中未被掩码（unmasked）的令牌取平均。
- 对每个序列中未被掩码的令牌求和，再除以一个常数标量（masked_normalize 函数支持通过设置 constant_normalizer≠1.0 实现）[Liu et al., 2025, Yu et al., 2025]。

假设批量大小为 2，第一个响应包含 4 个 tokens，第二个响应包含 7 个 tokens，通过以下示例可观察不同归一化方式对梯度的影响：

**问题：长度归一化思考（1分）**
交付要求：对比两种方法（暂不执行实验）。分析每种方法的优缺点，以及在哪些特定场景或示例中某一种方法更具优势？

设第 (i) 个样本的未mask token数为 (L_i)，逐token损失
$$
\ell_{i,t} = -A_i , r_{i,t},
$$

其中 (A_i) 是该序列共享的 advantage（沿长度广播），(r_{i,t}) 是逐token probability ratio（忽略裁剪）。

**方法A：`masked_mean`（按序列长度做平均）**

$$
\mathcal{L}^{\text{mean}}*i=\frac{1}{L_i}\sum*{t=1}^{L_i}\ell_{i,t}
= -A_i \cdot \frac{1}{L_i}\sum_{t=1}^{L_i} r_{i,t}
$$

优点：
- **“每条序列等权”**：不管输出长短，每个样本在总损失里大致占同一量级；批里长短差很多时更稳。
- **步长/尺度更稳定**：平均后梯度范数不随 (L_i) 线性增长，学习率更好调，方差通常更小。
- **抑制长度偏置**：不会因为“多写几个 token 就多累积梯度”而天然偏向更长的回答（尤其当 reward/advantage 是“整句一个分”时）。

缺点：
- **长序列的“总信用”被稀释**：如果一条长回复里有很多关键决策点（很多 token 都该被同一个 $A_i$ 强化），平均会把每个 token 的有效更新缩小到 $1/L_i$。
- **与“episode级目标”不完全一致**：当你把一条回复当成一次 episode、奖励是整段的回报时，RL 里更自然的是对该 episode 的 logprob 求和（对应下面的方法B），均值会改变这种等价性。

更适合的场景：
- **你希望“每个样本/每条回复贡献相近”**：如偏好学习里输出长度分布差异大、但不希望模型为了更大梯度而学会“变长”。
- **优势函数来自序列级打分且噪声较大**：均值可减小长序列的梯度爆炸与方差。
- **你更关心平均token质量**：例如生成风格、语气一致性、单token行为约束更强。

**方法B：`masked_normalize`（先求和，再除以常数 (C)）**

$$
\mathcal{L}^{\text{const}}*i=\frac{1}{C}\sum*{t=1}^{L_i}\ell_{i,t}
= -A_i \cdot \frac{1}{C}\sum_{t=1}^{L_i} r_{i,t}
$$

这里关键差异是：**对同一 $A_i$，梯度规模随 $L_i$ 近似线性增长**（若 (C) 固定）。

优点：
- **“按token数加权”**：长序列拥有更大总更新量，等价于“把一条长序列看成更多训练信号/更多决策点”。
- **更贴近 episode-logprob 的 RL 形式**：序列级优势乘以 $\sum_t \log \pi(a_t|s_t)$ 的结构更自然（尤其当 advantage 真的是该段整体好坏）。
- **可能更利于长文本任务的信用分配**：长回答往往需要持续多步保持正确轨迹，求和让这些多步都得到足够强的同向更新。

缺点：
- **长度偏置风险**：如果 reward/advantage 不是明确“按长度累积”的，长回复会天然更“重要”，模型可能学会变长来获得更大梯度影响（即使变长不提升奖励）。
- **训练不稳定/调参更敏感**：批内长度波动会直接导致梯度范数波动；学习率、clip 范围、KL 系数更难统一。
- **隐式改变“batch weighting”**：同样 batch size 下，长样本等价于被“重复采样更多次”（因为贡献更大）。

更适合的场景：
- **你把序列当作真正的 episode，且奖励与长度/步骤数更一致**：例如每步都有潜在代价/收益、或长序列确实包含更多需要被强化的决策。
- **长输出任务且你发现 `masked_mean` 让长回答学得太慢**：如长推理、长摘要、长对话一致性，平均会把信号摊薄。
- **你希望优化更接近“按token计的目标”**：比如你的评估/成本本身按 token 计（token 级预算、token 级正确率累计）。

接下来，我们将通过实证对比 `masked_mean`（掩码均值）与 `masked_normalize`（掩码归一化）。

**问题：长度归一化的影响（2分）（2个H100 GPU小时）**
交付要求 通过端到端的 GRPO 训练，对比基于 masked_mean 的归一化与 masked_normalize 的性能。提交验证集答案奖励曲线，对实验结果进行分析（包括其他呈现明显趋势的指标）。

提示：考虑与稳定性相关的指标，例如梯度范数（gradient norm）。 后续实验将采用性能更优的方案。

答：从图中 eval/correct 曲线数据可知，masked_mean 方案更好

**基于组标准差的归一化**
回顾 `compute_group_normalized_rewards`（组归一化奖励计算）的标准实现（基于 Shao 等人[2024]、DeepSeek-AI 等人[2025]的研究），该实现通过组标准差进行归一化。Liu 等人[2025]指出，除以组标准差可能会给训练过程引入不必要的偏差：标准差较低的问题（例如过易或过难的问题，其奖励值几乎全为 1 或全为 0）在训练中会获得更高的权重。

Liu等人[2025]提出移除基于标准差的归一化，我们已在 `compute_group_normalized_rewards` 中实现该方案，现需进行测试。

**问题：标准差归一化的影响（2分）（2个H100 GPU小时）**
交付要求 对比 `use_std_normalization == True`（启用标准差归一化）与 `use_std_normalization == False`（禁用标准差归一化）的性能。提交验证集答案奖励曲线，对实验结果进行分析（包括其他呈现明显趋势的指标）。

提示：考虑与稳定性相关的指标，例如梯度范数。后续实验将采用性能更优的组归一化方案。

答：从图中eval/correct曲线数据可知，`use_std_normalization == True` 方案更好

**离线策略（Off-policy）与在线策略（On-policy）**
目前我们实验所用的超参数均为在线策略（on-policy）：每个采样批次（rollout batch）仅执行一次梯度更新，因此除上述长度归一化和优势归一化的选择外，我们几乎完全采用了策略梯度的“原则性”近似 $\widehat{g}$。

该方法虽具有理论合理性和稳定性，但效率较低。采样过程需要通过策略模型生成响应，速度较慢，是 GRPO 训练的主要成本来源；而每个采样批次仅执行一次梯度更新可能不足以显著改变策略行为，显得较为浪费。

接下来我们将测试离线策略（off-policy）训练：每个 rollout batch 执行多次梯度更新（甚至多个epochs）。

**问题：实现离线策略 GRPO（Off-policy GRPO）**
交付要求 实现离线策略 GRPO 训练。

根据上述完整 GRPO 训练循环的实现情况，你可能已具备相关基础架构；若未具备，需完成以下实现：
1. 支持每个 `rollout batch` 执行多个轮次（epochs）的梯度更新，其中轮次数量和每个采样批次的优化器更新次数通过 `rollout_batch_size`（采样批次大小）、`epochs_per_rollout_batch`（每个采样批次的轮次）和 `train_batch_size`（训练批次大小）控制。
2. 修改主训练循环，在每个采样批次生成阶段之后、梯度更新内循环之前，从策略模型中获取响应的对数概率（logprobs），作为 `old_log_probs`（旧对数概率）。建议使用 `torch.inference_mode()`（推理模式）提升效率。
3. 采用 GRPO-Clip 损失类型。
通过调整每个采样批次的轮次数量和优化器更新次数，可控制离线策略的程度。

**问题：离线策略 GRPO 超参数搜索（4分）（12个H100 GPU小时）**
交付要求
固定 `rollout_batch_size = 256`（采样批次大小=256），选择 `epochs_per_rollout_batch`（每个采样批次的轮次）和 `train_batch_size`（训练批次大小）的搜索范围。首先进行大范围搜索（GRPO 步数 < 50）以初步了解性能分布，再进行聚焦搜索（GRPO 步数=200）。提交简要实验日志，说明所选搜索范围的依据。

与在线策略实验（`epochs_per_rollout_batch = 1`、`train_batch_size = 256`）进行对比，提交以验证步数和壁钟时间（wall-clock time）为横轴的对比图。

提交验证集答案奖励曲线，对实验结果进行分析（包括其他呈现明显趋势的指标，如熵（entropy）和响应长度）。对比训练过程中模型响应的熵与EI实验中的观察结果。
提示：需调整 `gradient_accumulation_steps`（梯度累积步数）以保持显存使用量稳定。

答：`epochs_per_rollout_batch` 的搜索范围：[2, 3, 4]; `train_batch_size` 的搜索范围：[32, 64, 128, 256]; n_grpo_steps=200.

当 `epochs_per_rollout_batch` 固定为2时，`train_batch_size=256` 曲线的 reward 最高。当 `epochs_per_rollout_batch` 固定为3时，`train_batch_size=256` 曲线的 reward 最高。当 `epochs_per_rollout_batch` 固定为4时，`train_batch_size=64` 曲线的 reward 值最高，但和前两个曲线相比，是效果非常差的超参数。

**离线策略下的裁剪（Clipping）消融实验**
回顾 GRPO-Clip 中裁剪（clipping）的目的：当对单个采样批次执行多次梯度更新时，防止策略与旧策略（old policy）偏离过大。接下来，我们将在离线策略设置中消融裁剪操作，测试其必要性——即采用逐 token（per-token）损失。

**问题：离线策略 GRPO-Clip 消融实验（2分）（2个H100 GPU小时）**
交付要求
实现无裁剪的逐 token 损失，作为新的损失类型“GRPO-No-Clip”（无裁剪GRPO）。使用上一个问题中性能最优的离线策略超参数，运行无裁剪版本的损失函数。提交验证集答案奖励曲线，与 GRPO-Clip 的实验结果对比分析（包括其他呈现明显趋势的指标，如熵、响应长度和梯度范数）。

答：前文实验中， 最优超参数 epochs_per_rollout_batch=2，train_batch_size=256。发现在这组参数下，clip-fraction 曲线都是0，因此选择次优超参数 epochs_per_rollout_batch=3，train_batch_size=256。

eval/correct 曲线说明，no-clip 并不一定就不如 clip。在离线 GRPO、奖励稀疏且 token-level advantage 集中的设置下， No-Clip 并不一定劣于 Clip。当 policy 更新本身处于低风险区间时，Clip 可能反而限制了有效学习信号的利用。

**提示词（Prompt）的影响**
作为最后一项消融实验，我们将研究一个有趣的现象：基于模型的预训练方式，强化学习（RL）过程中使用的提示词可能会对模型性能产生显著影响。

替代 cs336_alignment/prompts/r1_zero.prompt 路径下的 R1-Zero 提示词，使用 cs336_alignment/prompts/question_only.prompt 路径下的极简提示词：

```
{question}  # 仅包含问题本身
```

该提示词将用于训练和验证阶段，并将奖励函数（训练和验证均使用）修改为 cs336_alignment/drgrpo_grader.py 中的 question_only_reward_fn（仅问题奖励函数）。

**问题：提示词消融实验（2分）（2个H100 GPU小时）**
交付要求
提交 R1-Zero 提示词与仅问题提示词的验证集答案奖励曲线。对比两种提示词下的各项指标（包括熵、响应长度和梯度范数等呈现明显趋势的指标），并尝试解释实验结果。
答：从 eval 曲线来看，仅问题提示词效果效果非常差，但是训练曲线上的效果却很好，很能迷惑人。

