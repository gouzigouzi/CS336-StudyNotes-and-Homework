import torch


def save_checkpoint(model, optimizer, iteration, out):
    """
    保存模型检查点
    
    参数:
        model: 要保存的模型
        optimizer: 优化器
        iteration: 当前训练迭代次数
        out: 保存目标 (文件路径/文件对象)
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)

def load_checkpoint(src, model, optimizer):
    """
    加载模型检查点
    
    参数:
        src: 检查点源 (文件路径/文件对象)
        model: 待恢复的模型
        optimizer: 待恢复的优化器
    
    返回:
        保存的迭代次数
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration