# The problem occurs because the normalization transforms 
# are computed on CPU but the model runs on GPU.
# Here's the fixed code for the GPU training cell:

import torch
import torch_directml
import time
from fastai.vision.all import *

def fix_gpu_training_error(dls_base, resnet18, accuracy):
    """
    Fix for the GPU training error where tensors are on different devices
    """
    gpu_device = torch_directml.device()
    print('DirectML device:', torch_directml.device_name(0))

    # Create GPU dataloaders with the correct device
    dls_gpu = DataLoaders(
        dls_base.train.new(device=gpu_device),
        dls_base.valid.new(device=gpu_device),
        path=dls_base.path,
        device=gpu_device
    )

    # Create the model with GPU dataloaders
    learn_gpu = cnn_learner(dls_gpu, resnet18, metrics=accuracy)
    
    # Ensure model is on the GPU
    learn_gpu = learn_gpu.to(gpu_device)

    # Make sure not to use fp16 if it causes issues with DirectML
    # If the to_fp16 method exists and works with DirectML, use it
    # Otherwise, skip it to avoid the device mismatch issue
    try:
        learn_gpu = learn_gpu.to_fp16(enabled=False)
    except Exception as e:
        print(f"Warning: Could not apply fp16: {e}")
        pass

    n_epochs = 2
    start = time.perf_counter()
    
    # This should now work without the device mismatch error
    learn_gpu.fit_one_cycle(n_epochs, lr_max=3e-3)
    
    gpu_time = time.perf_counter() - start

    print(f'GPU DirectML elapsed: {gpu_time:.2f}s')
    
    return learn_gpu, gpu_time

# Alternative simpler fix: Make sure normalization stats are on the right device
def simple_fix(dls_base, resnet18, accuracy):
    """
    A simpler fix that ensures normalization is applied correctly
    """
    gpu_device = torch_directml.device()
    print('DirectML device:', torch_directml.device_name(0))

    # Ensure all normalization happens on the GPU
    dls_gpu = dls_base.cuda() if hasattr(dls_base, 'cuda') else dls_base
    dls_gpu = dls_gpu.new(device=gpu_device)
    
    # Create the model
    learn_gpu = cnn_learner(dls_gpu, resnet18, metrics=accuracy)
    
    # Move everything to GPU explicitly
    learn_gpu.dls.device = gpu_device
    learn_gpu = learn_gpu.to(gpu_device)

    n_epochs = 2
    start = time.perf_counter()
    learn_gpu.fit_one_cycle(n_epochs, lr_max=3e-3)
    gpu_time = time.perf_counter() - start

    print(f'GPU DirectML elapsed: {gpu_time:.2f}s')
    
    return learn_gpu, gpu_time