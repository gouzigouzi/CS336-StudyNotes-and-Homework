import torch
import torch.distributed as dist
from torch.autograd.profiler import record_function

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super(DDP, self).__init__()
        self.module = module
        self.handles = []

        # 初始参数广播：确保所有进程都有相同的初始参数
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.transform_grad)  # 注册一个hook，在每次梯度累积后调用transform_grad函数

    def transform_grad(self, param):
        with torch.no_grad():
            param.grad.data /= dist.get_world_size()  # 平均化梯度

        with record_function("allreduce_async"):
            self.handles.append(dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True))  # 异步地进行all-reduce操作，将梯度在所有进程之间进行平均化

    def finish_gradient_synchronization(self):
        # 等待所有的all-reduce操作完成，确保梯度已经同步
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
    
    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)
