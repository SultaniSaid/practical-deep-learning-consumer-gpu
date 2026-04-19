"""
Solution for DirectML GPU training error in lesson1_gpu_vs_cpu.ipynb

This script fixes the "Expected all tensors to be on the same device" error
by ensuring all model components and data are on the same device.
"""

import torch
import torch_directml
import time
from fastai.vision.all import *

def fix_gpu_training_error():
    """
    This function demonstrates the fix for the device mismatch error
    """
    print("Applying fix for DirectML GPU training...")
    
    # First, recreate the original dataloaders if needed
    # (Assuming these are available in the notebook environment)
    img_path = Path('images')  # Update this path accordingly
    pat = r'([^/]+)_\d+.jpg$'
    
    def label_func(o): 
        # Processing string or Path object
        if hasattr(o, 'name'):
            return o.name.split('_')[0]
        else:
            import ntpath
            basename = ntpath.basename(o)
            return basename.split('_')[0]

    # Recreate base dataloader with no workers to prevent potential issues
    dls_base = ImageDataLoaders.from_name_func(
        img_path, get_image_files(img_path), label_func,
        valid_pct=0.2, seed=42,
        item_tfms=Resize(128), bs=32,
        num_workers=0  # Using 0 workers might help with DirectML compatibility
    )
    
    # Get the GPU device
    gpu_device = torch_directml.device()
    print('DirectML device:', torch_directml.device_name(0))
    
    # Create GPU-specific dataloaders
    dls_gpu = DataLoaders(
        dls_base.train.new(device=gpu_device),
        dls_base.valid.new(device=gpu_device),
        path=dls_base.path,
        device=gpu_device,
        bs=dls_base.bs
    )
    
    # Create the model with GPU dataloaders
    learn_gpu = cnn_learner(dls_gpu, resnet18, metrics=accuracy)
    
    # Ensure the model is moved to GPU
    learn_gpu = learn_gpu.to(gpu_device)
    
    # Skip fp16 conversion which might cause issues with DirectML
    n_epochs = 2
    start = time.perf_counter()
    
    # Train the model
    learn_gpu.fit_one_cycle(n_epochs, lr_max=3e-3)
    gpu_time = time.perf_counter() - start
    
    print(f'GPU DirectML elapsed: {gpu_time:.2f}s')
    return learn_gpu, gpu_time


def alternative_fix_for_existing_dataloaders(dls_base):
    """
    Apply fix to existing dataloaders (when dls_base is already created)
    """
    gpu_device = torch_directml.device()
    print('DirectML device:', torch_directml.device_name(0))
    
    # Ensure all tensors go to the correct device by recreating the DataLoader objects
    # This forces the normalization and other transforms to happen on the correct device
    train_dl = dls_base.train.new(after_item=[ToTensor(), IntToFloatTensor()])
    valid_dl = dls_base.valid.new(after_item=[ToTensor(), IntToFloatTensor()])
    
    dls_gpu = DataLoaders(train_dl, valid_dl, device=gpu_device)
    
    # Create the model
    learn_gpu = cnn_learner(dls_gpu, resnet18, metrics=accuracy).to(gpu_device)
    
    n_epochs = 2
    start = time.perf_counter()
    learn_gpu.fit_one_cycle(n_epochs, lr_max=3e-3)
    gpu_time = time.perf_counter() - start
    
    print(f'GPU DirectML elapsed: {gpu_time:.2f}s')
    return learn_gpu, gpu_time


if __name__ == "__main__":
    print("Running DirectML GPU training fix...")
    try:
        # Try the main fix
        model, time_taken = fix_gpu_training_error()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Try using the alternative fix with existing dataloaders")