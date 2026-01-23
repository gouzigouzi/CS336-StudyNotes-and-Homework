from torch import optim
import torch

class AdamW(optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # print(state)
                # 初始化状态缓存：step, 动量 m, RMS v
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)  # 动量, 一阶矩估计
                    state['v'] = torch.zeros_like(p.data)  # RMS, 二阶矩估计
                
                m, v = state['m'], state['v']  # 这只是引用罢了，也是改变原始状态的
                beta1, beta2 = group['betas']

                state['step'] += 1    

                # 更新动量：m = beta1 * m + (1 - beta1) * grad
                m.mul_(beta1).add_(grad, alpha=1 - beta1)  # mul_ 和 add_ 都是inplace操作，就地修改
                # 更新 RMS：v = beta2 * v + (1 - beta2) * grad^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # 偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1 
                
                # 分母：sqrt(v / bias_correction2) + eps
                denom = (v / bias_correction2).sqrt().add_(group['eps']) 
                # 参数更新：p = p - step_size * (m / denom)
                p.data.addcdiv_(-step_size, m, denom)  # 先乘后除

                # 解耦权重衰减,直接减，不要修改梯度
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])