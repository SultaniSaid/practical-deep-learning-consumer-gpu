import torch
import torch_directml
import time
from fastai.vision.all import *

# 假设前面的代码已经定义了 dls_base、resnet18 和 accuracy
# 以下是修复后的GPU训练代码

def train_with_gpu_fix():
    gpu_device = torch_directml.device()
    print('DirectML device:', torch_directml.device_name(0))
    
    # 确保数据加载器使用正确的设备
    dls_gpu = DataLoaders(
        dls_base.train.new(device=gpu_device),
        dls_base.valid.new(device=gpu_device),
        path=dls_base.path,
        device=gpu_device
    )
    
    # 创建模型时指定设备
    learn_gpu = cnn_learner(dls_gpu, resnet18, metrics=accuracy).to_fp16(enabled=False)
    
    # 确保整个模型被移动到GPU
    learn_gpu.dls.device = gpu_device
    for m in learn_gpu.modules():
        m.to(gpu_device)
        
    # 显式地将模型移动到目标设备
    learn_gpu = learn_gpu.to(gpu_device)
    
    n_epochs = 2
    start = time.perf_counter()
    
    # 执行训练
    learn_gpu.fit_one_cycle(n_epochs, lr_max=3e-3)
    
    gpu_time = time.perf_counter() - start
    
    print(f'GPU DirectML elapsed: {gpu_time:.2f}s')
    
    return learn_gpu, gpu_time

if __name__ == "__main__":
    train_with_gpu_fix()