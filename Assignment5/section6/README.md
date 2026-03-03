# CS336 Assignment 5 (alignment): Alignment and Reasoning RL

## 6 策略梯度入门( Primer on Policy Gradients)
语言模型研究中的一项令人振奋的新发现是：利用性能强劲的基础模型，针对经过验证的奖励信号进行强化学习（RL），能够显著提升模型的推理能力和性能[OpenAI等，2024；DeepSeek-AI等，2025]。目前性能最优异的开源推理模型（如DeepSeek R1和Kimi k1.5[团队等，2025]）均采用策略梯度算法（policy gradients）训练而成。策略梯度是一种强大的强化学习算法，可优化任意奖励函数。

下文将简要介绍适用于语言模型强化学习的策略梯度方法。本部分内容主要基于两份优质参考资料（对相关概念有更深入的阐述）：OpenAI的《深度强化学习入门》[Achiam，2018a]和内森·兰伯特（Nathan Lambert）的《基于人类反馈的强化学习（RLHF）手册》[Lambert，2024]。

### 6.1 将语言模型当作策略
参数为 $\theta$ 的因果语言模型（LM）定义了一个概率分布：给定当前文本前缀 $s_t$（状态/观测值），下一个 token $a_t \in V$（$V$ 为词汇表）的出现概率。在强化学习语境中，我们将下一个 token $a_t$ 视为“动作”，将当前文本前缀 $s_t$ 视为“状态”。因此，语言模型可看作一种类别型随机策略( categorical stochastic policy)。
$$
a_t \sim \pi_\theta(\cdot | s_t), \quad \pi_\theta(a_t | s_t) = [\text{softmax}(f_\theta(s_t))]_{a_t}
$$

通过策略梯度优化策略时，需用到两项基础操作：
1. 从策略中采样：从上述类别分布中抽取一个动作 $a_t$；
2. 计算动作的对数似然：评估 $\log \pi_{\theta}(a_t | s_t)$（给定状态 $s_t$ 时，策略 $\pi_{\theta}$ 选择动作 $a_t$ 的对数概率）。

通常来说，在大语言模型（LLMs）的强化学习任务中，$s_t$ 指的是目前生成的部分完成内容/解决方案，而每个 $a_t$ 是该解决方案的下一个 token；当输出文本结束 token（比如 $<| \text{end\_of\_text} |>$）时，该轮交互就会结束；在我们的 r1_zero 提示词场景下，对应的结束token是 $</\text{answer}>$。

### 6.2 轨迹（Trajectories）
（有限时域的）轨迹是 agent 经历的状态与动作的交替序列：
$$\tau = \left(s_0, a_0, s_1, a_1, \dots, s_T, a_T\right)$$

其中 $T$ 为轨迹长度，即 $a_T$ 是文本结束 token，或已达到最大生成 token 数上限。

初始状态 state 从起始分布中采样得到：$ s_0 \sim \rho_0(s_0)$；在语言模型的强化学习中，$\rho_0(s_0)$ 是 formatted prompts 的分布。在一般场景中，状态转移遵循 environment dynamics $s_{t+1} \sim P(\cdot | s_t, a_t)$。而在语言模型的强化学习中，环境是确定性的：下一个状态是旧前缀与生成 token 的拼接，即 $s_{t+1} = s_t \| a_t$（“$\|$”表示字符串拼接）。轨迹（Trajectories）也被称为“轮次（episodes）”或“滚动（rollouts）”，本文中这三个术语可互换使用。

### 6.3 奖励与回报（Rewards and Return）
标量奖励 $r_t = R(s_t, a_t)$ 用于评判在状态 $s_t$ 下所执行动作的即时优劣。在经过验证的领域（如数学解题）中，强化学习的标准做法是：中间步骤的奖励设为 0，终端动作的奖励为经过验证的结果，即：
$$
r_T = R(s_T, a_T) := 
\begin{cases} 
1 & \text{若 } s_T \| a_T \text{ 与奖励函数定义的真实结果一致} \\ 
0 & \text{否则} 
\end{cases}
​$$

回报 $R(\tau)$ 是轨迹上所有奖励的累加。两种常见定义为：有限时域无折扣回报：
$$
R(\tau) := \sum_{t=0}^T r_t
$$

- 含义：将从开始（第 $0$ 步）到明确的结束点（第 $T$ 步）的所有每步奖励 $r_t$ 直接相加。
- 特点：任务有明确的终点（例如文本生成结束）。无论奖励是早期获得还是晚期获得，其权重都是相同的（即“无折扣”）。在本次作业的数学解题任务中，使用的是这种回报，因为回答过程有自然的结束点。

无限时域折扣回报：
$$
R(\tau) := \sum_{t=0}^{\infty} \gamma^t r_t, \quad 0 < \gamma < 1
$$

- 含义：假设任务没有明确的终点（时域为无限的），将未来的所有奖励相加。
- 特点：为了防止总和趋于无穷大，引入了折扣因子 $\gamma$（取值在 0 到 1 之间）。由于 $\gamma^t$ 会随着时间 $t$ 的增加而变小，这意味着越晚获得的奖励，其现在的价值越低。这促使模型更倾向于快速获得奖励（即看重短期收益）。

在本实验中，由于交互轮次有自然终止点（文本结束或最大生成长度），我们将使用有限时域无折扣回报定义。

agent 的目标是最大化期望回报：
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

对应的优化问题为：
$$\theta^* = \arg \max_{\theta} J(\theta)$$

### 6.4 vanilla 策略梯度（Vanilla Policy Gradient）
接下来，我们尝试通过梯度上升法最大化期望回报，从而学习策略参数 $\theta$：
$$\theta_{k+1} = \theta_k + \alpha \nabla_\theta J(\theta_k)$$

核心公式为 REINFORCE 策略梯度（如下所示），它是实现这一目标的关键：
$$\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t | s_t) R(\tau)\right]. \tag{10}$$

梯度 $\nabla_{\theta} J(\pi_{\theta})$ 表示期望回报随策略参数 $\theta$ 的变化率，期望 $\mathbb{E}_{\tau \sim \pi_{\theta}}$ 表示对所有从策略 $\pi_{\theta}$ 采样的轨迹 $\tau$ 取平均。

**策略梯度的推导**
该公式如何推导而来？为保证完整性，下文将简要推导。推导过程将用到以下几项结论：

轨迹的概率为：
$$P(\tau \mid \theta) = \rho_0(s_0) \prod_{t=0}^T P(s_{t+1} \mid s_t, a_t) \pi_\theta(a_t \mid s_t).$$

因此，轨迹的对数概率为：
$$\log P(\tau \mid \theta) = \log \rho_0(s_0) + \sum_{t=0}^T \left[ \log P(s_{t+1} \mid s_t, a_t) + \log \pi_\theta(a_t \mid s_t) \right].$$

对数导数技巧（log-derivative trick）：
$$\nabla_\theta P = P \nabla_\theta \log P.$$

环境相关项与 $\theta$ 无关：$\rho_0$ 、$P(\cdot | \cdot)$ 和 $R(\tau)$ 均不依赖于策略参数，因此对 $\theta$ 的梯度为 0。
$$\nabla_\theta \rho_0 = \nabla_\theta P = \nabla_\theta R(\tau) = 0.$$

基于上述公式：
$$\begin{align*}
\nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] \\
&= \nabla_\theta \sum_\tau P(\tau \mid \theta) R(\tau) \\
&= \sum_\tau \nabla_\theta P(\tau \mid \theta) R(\tau) \\
&= \sum_\tau P(\tau \mid \theta) \nabla_\theta \log P(\tau \mid \theta) R(\tau) \quad (\text{Log-derivative trick}) \\
&= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau \mid \theta) R(\tau) \right].
\end{align*}
$$

将轨迹的对数概率代入，并利用“环境项与 $\theta$ 无关”这一性质，即可得到 vanilla 或 REINFORCE 策略梯度公式：
$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) R(\tau) \right].$$

直观来看，该梯度会增大“高回报轨迹中所有动作”的对数概率，同时减小“低回报轨迹中所有动作”的对数概率。

**梯度的采样估计**
给定一批由 $N$ 条轨迹组成的数据集 $D = \{\tau^{(i)}\}_{i=1}^N$（采样方式：从起始分布 $\rho_0(s_0)$ 中抽取初始状态 $s_0^{(i)}$，然后在环境中运行策略 $\pi_{\theta}$ 生成轨迹），可构造梯度的无偏估计：
$$\hat{g} = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \nabla_\theta \log \pi_\theta \left( a_t^{(i)} \mid s_t^{(i)} \right) R\left( \tau^{(i)} \right).$$

该向量将用于梯度上升更新：$\theta \leftarrow \theta + \alpha \hat{g}$（其中 $\alpha$ 为学习率）

### 6.5 策略梯度的基线（Policy Gradient Baselines）
vanilla 策略梯度的主要问题是梯度估计的方差较大。一种常用的缓解方法是在奖励中减去一个仅依赖于状态的基线函数 $b$。这是一种控制变量法 (control variate) [Ross, 2022]：核心思想是通过减去一个与梯度估计相关的项，在不引入偏差的前提下降低估计方差。

定义带基线的策略梯度为：
\[
B = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta \left( a_t \mid s_t \right) \left( R(\tau) - b(s_t) \right) \right].
\]

例如，一个合理的基线是“在策略价值函数（on-policy value function）” \( V^\pi(s) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \mid s_t = s \right] \)，即：从状态 \( s_t = s \) 出发并遵循策略 \( \pi_\theta \) 时的期望回报。此时，\( (R(\tau) - V^\pi(s)) \) 直观上表示“实际轨迹回报与期望回报的差值”（即优势值）。

只要基线仅依赖于状态，带基线的策略梯度就是无偏的。这一点可通过重写带基线的策略梯度证明：

\[
B = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta \left( a_t \mid s_t \right) R(\tau) \right] - \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta \left( a_t \mid s_t \right) b(s_t) \right].
\]

聚焦基线项，我们发现：

\[
\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta \left( a_t \mid s_t \right) b(s_t) \right] = \sum_{t=0}^T \mathbb{E}_{s_t} \left[ b(s_t) \mathbb{E}_{a_t \sim \pi_\theta \mid s_t} \left[ \nabla_\theta \log \pi_\theta \left( a_t \mid s_t \right) \right] \right].
\]

一般而言，得分函数的期望为 0：\( \mathbb{E}_{x \sim P_\theta} \left[ \nabla_\theta \log P_\theta(x) \right] = 0 \)。因此，上式中的基线项期望为 0，即：

\[
B = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta \left( a_t \mid s_t \right) R(\tau) \right] - 0 = \nabla_\theta J(\pi_\theta).
\]

由此可得出结论：带基线的策略梯度是无偏的。后续我们将通过实验验证基线是否能提升下游任务性能。

**关于策略梯度“损失函数”的说明**
在 PyTorch 等框架中实现策略梯度方法时，我们会定义一个所谓的“策略梯度损失”（pg_loss），使得调用 `pg_loss.backward()` 时，模型参数的梯度缓冲区会被填充为近似策略梯度 $\hat{g}$。从数学上看，其定义满足：
$$
\text{pg\_loss} = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \log \pi_\theta\left(a_t^{(i)} \mid s_t^{(i)}\right)\left(R\left(\tau^{(i)}\right) - b\left(s_t^{(i)}\right)\right).
$$

但 pg_loss 并非传统意义上的“损失函数”——将其作为训练集或验证集的评估指标是无意义的，良好的验证集 pg_loss 并不代表模型具有良好的泛化能力。pg_loss 本质上只是一个标量，其核心作用是：通过反向传播（backprop）得到近似策略梯度 $\hat{g}$。

在强化学习中，**应始终记录并报告训练集和验证集的“奖励”（rewards）**。奖励是真正有意义的评估指标，也是我们通过策略梯度方法试图优化的目标。

### 6.6 异策略策略梯度（Off-Policy Policy Gradient）
REINFORCE 是一种“同策略（on-policy）”算法：训练数据由当前正在优化的策略生成，用来更新模型的训练数据（轨迹），完全是由当前时刻、正在被优化的这个模型生成的。这一点可通过 REINFORCE 算法的步骤明确看出：
1. 从当前策略 $\pi_{\theta}$ 中采样一批轨迹 $\{\tau^{(i)}\}_{i=1}^N$；
2. 近似策略梯度：$\nabla_{\theta} J(\pi_{\theta}) \approx \hat{g} = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t^{(i)} | s_t^{(i)}) R(\tau^{(i)})$；
3. 利用计算得到的梯度更新策略参数：$\theta \leftarrow \theta + \alpha \hat{g}$。

该方法的问题在于：需要进行大量推理以采样新的轨迹批次，却仅能执行一步梯度更新。由于语言模型的行为通常无法在单步更新中发生显著变化，这种同策略方法的效率极低。

**异策略策略梯度**
在异策略学习中，轨迹采样自“非当前优化的策略”，也就是“从过去的经验中学”，用来更新当前模型的数据，是由过去旧版本的模型（或者完全不同的其他模型 $\pi_{\theta_{old}}$）生成的。主流策略梯度算法（如 PPO、GRPO）的异策略变体，会利用“旧策略 $\pi_{\theta_{old}}$ 生成的轨迹”来优化当前策略 $\pi_{\theta}$。异策略的策略梯度估计为：
$$\widehat{g}_{\text{off-policy}} = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T \frac{\pi_\theta\left(a_t^{(i)} \mid s_t^{(i)}\right)}{\pi_{\theta_{\text{old}}}\left(a_t^{(i)} \mid s_t^{(i)}\right)} \nabla_\theta \log \pi_\theta\left(a_t^{(i)} \mid s_t^{(i)}\right) R\left(\tau^{(i)}\right). \tag{27}$$

该式可看作 vanilla 策略梯度的**重要性采样**版本，其中包含了 $\frac{\pi_{\theta}(a_t^{(i)} | s_t^{(i)})}{\pi_{\theta_{\text{old}}}(a_t^{(i)} | s_t^{(i)})}$ 这样的重加权项。实际上，上式可通过重要性采样推导得出，且在 $\pi_{\theta}$ 与 $\pi_{\theta_{old}}$ 差异不大的前提下，该近似是合理的（更多细节参见 Degris 等 [2013]）。由于数据是旧模型生成的，直接用来评估当前模型会有偏差。引入了**重要性采样**机制，即公式中的重加权项 $\frac{\pi_{\theta}(a_t^{(i)} | s_t^{(i)})}{\pi_{\theta_{\text{old}}}(a_t^{(i)} | s_t^{(i)})}$。它通过比对新旧策略输出同一动作的概率差，来修正梯度的方向。该方法大大提升了数据的利用率，同一批数据可以反复用来更新当前模型。
