import torch
import torch_directml
import logging
import os, warnings
from pathlib import Path
from fastai.vision.all import Normalize as VisionNormalize
from fastai.tabular.all import Normalize as TabularNormalize, FillMissing
from fastai.text.all import TensorText, show_title
from fastai.layers import AdaptiveConcatPool2d
from fastai.learner import Learner
from fastai.callback.fp16 import MixedPrecision
from fastai.basics import patch, Module, Callback

# ==============================================================================
# DIRECTML VANGUARD: GLOBAL AUTO-PATCHES (v3.5)
# These patches apply automatically upon import to ensure training stability.
# ==============================================================================

# 1. Device Registry
CURRENT_DEVICE = torch_directml.device() # Default to GPU

# 2. Patch Normalize (Dynamic Device Sync)
if not hasattr(VisionNormalize, '_dml_patched'):
    old_vision_init = VisionNormalize.__init__
    def new_vision_init(self, mean, std, axes=(0,2,3)):
        old_vision_init(self, mean.to(CURRENT_DEVICE), std.to(CURRENT_DEVICE), axes=axes)
    VisionNormalize.__init__ = new_vision_init
    VisionNormalize._dml_patched = True

# 3. Patch Learner to FORCE Deep Sync & Gradients (THE CORE FIX)
if not hasattr(Learner, '_dml_device_patched'):
    old_learner_init = Learner.__init__
    def new_learner_init(self, *args, **kwargs):
        old_learner_init(self, *args, **kwargs)
        if hasattr(self, 'dls') and hasattr(self, 'model'):
            # Force model to correct device and RE-ENABLE gradients
            self.model.to(self.dls.device)
            for p in self.model.parameters(): 
                p.requires_grad_(True)
            # Re-sync optimizer to the new parameter memory space
            if hasattr(self, 'opt') and self.opt is not None:
                self.create_opt() 
    Learner.__init__ = new_learner_init
    Learner._dml_device_patched = True

# 4. Universal Pooling Patch (DML Safe + Gradients)
if not hasattr(AdaptiveConcatPool2d, '_dml_patched'):
    import torch.nn as nn
    def concat_pool_forward(self, x):
        ap = nn.AdaptiveAvgPool2d(self.size)(x)
        mp = nn.AdaptiveMaxPool2d(self.size)(x)
        return torch.cat([ap, mp], 1)
    AdaptiveConcatPool2d.forward = concat_pool_forward
    AdaptiveConcatPool2d._dml_patched = True

# 5. Patch MixedPrecision to DISABLE for DirectML stability
if not hasattr(MixedPrecision, '_dml_patched'):
    def no_op(self, *args, **kwargs): return
    for hook in ['before_fit', 'before_batch', 'after_pred', 'after_loss', 
                 'before_backward', 'before_step', 'after_step', 'after_fit']:
        setattr(MixedPrecision, hook, no_op)
    MixedPrecision._dml_patched = True

# ==============================================================================
# USER-FACING UTILITIES
# ==============================================================================

def setup_dml(device_name=None):
    """
    Sets up the compute device and prints system info.
    """
    global CURRENT_DEVICE
    
    if device_name is None or device_name == 'gpu':
        CURRENT_DEVICE = torch_directml.device()
        print("="*80)
        print("DIRECTML VANGUARD MODULE LOADED (v3.5 AUTO-SYNC)")
        print(f"Device: {CURRENT_DEVICE}")
        print(f"GPU: {torch_directml.device_name(0)}")
        print("="*80)
    else:
        CURRENT_DEVICE = torch.device('cpu')
        print("="*80)
        print("DIRECTML VANGUARD: RUNNING ON CPU")
        print("="*80)
    
    # Silence redundant CUDA/AMP warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*CUDA is not available.*")
    return CURRENT_DEVICE

def get_local_path():
    p = Path('data')
    p.mkdir(parents=True, exist_ok=True)
    return p

def optimize_dls(dls):
    """Applies stability settings for Adreno/Windows"""
    # Force cpu for splitting, then pin to DML
    dls.to(CURRENT_DEVICE)
    # Ensure workers are stable
    if dls.num_workers > 4: dls.num_workers = 4
    return dls
