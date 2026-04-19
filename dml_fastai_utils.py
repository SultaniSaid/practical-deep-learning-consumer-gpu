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

def setup_dml():
    """
    Sets up the DirectML device and returns it.
    Also applies global patches to FastAI classes (v3.0 Performance Vanguard).
    """
    dml = torch_directml.device()
    
    # Silence redundant CUDA/AMP warnings on DirectML/Adreno
    warnings.filterwarnings("ignore", category=UserWarning, message=".*CUDA is not available.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.cuda.amp.GradScaler is enabled.*")

    print("="*80)
    print("DIRECTML VANGUARD MODULE LOADED (v3.0)")
    print(f"Device: {dml}")
    print(f"GPU: {torch_directml.device_name(0)}")
    print("="*80)
    
    # 1. Patch Normalize for both Vision and Tabular
    old_vision_init = VisionNormalize.__init__
    def new_vision_init(self, mean, std, axes=(0,2,3)):
        old_vision_init(self, mean.to(dml), std.to(dml), axes=axes)
    VisionNormalize.__init__ = new_vision_init

    old_tabular_setups = TabularNormalize.setups
    def new_tabular_setups(self, to):
        old_tabular_setups(self, to)
        if hasattr(self, 'mean') and self.mean is not None:
            self.mean = self.mean.to(dml)
        if hasattr(self, 'std') and self.std is not None:
            self.std = self.std.to(dml)
    TabularNormalize.setups = new_tabular_setups

    # 2. Patch FillMissing for Tabular
    old_fillmissing_setups = FillMissing.setups
    def new_fillmissing_setups(self, to):
        old_fillmissing_setups(self, to)
        if hasattr(self, 'na_dict') and self.na_dict:
            for k, v in self.na_dict.items():
                if isinstance(v, torch.Tensor):
                    self.na_dict[k] = v.to(dml)
    FillMissing.setups = new_fillmissing_setups

    # 3. Patch Learner.freeze_to for DirectML safety
    def directml_safe_freeze_to(self: Learner, n: int):
        if hasattr(self.model, 'hf_model'):
            if n == -1:
                for p in self.model.hf_model.base_model.parameters(): p.requires_grad_(False)
                if hasattr(self.model.hf_model, 'classifier'):
                    for p in self.model.hf_model.classifier.parameters(): p.requires_grad_(True)
            else:
                for p in self.model.parameters(): p.requires_grad_(True)
        else:
            if n == 0:
                for p in self.model.parameters(): p.requires_grad_(True)
            else:
                backbone = self.model[0] if hasattr(self.model, '__getitem__') else self.model
                backbone_ids = {id(p) for p in backbone.parameters()}
                for p in backbone.parameters(): p.requires_grad_(False)
                for p in self.model.parameters():
                    if id(p) not in backbone_ids: p.requires_grad_(True)
        if self.opt is not None: self.opt.clear_state()
            
    Learner.freeze_to = directml_safe_freeze_to
    
    # 4. Patch TensorText
    @patch
    def truncate(self: TensorText, n): return type(self)(self[:n])

    @patch  
    def show(self: TensorText, ctx=None, **kwargs):
        tokenizer = kwargs.get('tokenizer', getattr(self, '_tokenizer', None))
        text = tokenizer.decode(self.cpu().tolist(), skip_special_tokens=True) if tokenizer else str(self.cpu().tolist())
        if ctx is None: print(text); return text
        return show_title(text, ctx=ctx, **kwargs)

    # 5. Patch AdaptiveConcatPool2d (High-Performance Pooling)
    # This prevents 'aten::adaptive_max_pool2d.out' CPU fallback by replacing MaxPool with a fast amax reduction
    def dml_safe_concat_pool_forward(self, x):
        return torch.cat([self.ap(x), torch.amax(x, dim=(2,3), keepdim=True)], 1)
    AdaptiveConcatPool2d.forward = dml_safe_concat_pool_forward

    # 6. Patch MixedPrecision to avoid CUDA-centric defaults
    old_mp_init = MixedPrecision.__init__
    def new_mp_init(self, *args, **kwargs):
        if 'device_type' not in kwargs:
            # Force 'cpu' for AMP context if CUDA is missing, avoiding 'cuda' warnings
            kwargs['device_type'] = 'cpu'
        old_mp_init(self, *args, **kwargs)
    MixedPrecision.__init__ = new_mp_init

    return dml

def optimize_dls(dls, num_workers=8, pin_memory=False, persistent_workers=True):
    """Optimizes DataLoaders for DirectML performance on high-core CPUs."""
    dls.train.n_workers = num_workers
    dls.valid.n_workers = num_workers
    dls.train.pin_memory = pin_memory
    dls.valid.pin_memory = pin_memory
    
    # Try to set persistent_workers if using multiple workers
    if num_workers > 0:
        try:
            dls.train.persistent_workers = persistent_workers
            dls.valid.persistent_workers = persistent_workers
        except: pass

    print(f"DataLoaders optimized: num_workers={num_workers}, pin_memory={pin_memory}, persistent={persistent_workers}")
    return dls

class HFCallback(Callback):
    def __init__(self, pad_idx): self.pad_idx = pad_idx
    def before_batch(self):
        xb = self.xb[0] if isinstance(self.xb, tuple) and len(self.xb) > 0 else self.xb
        self.model._attention_mask = (xb != self.pad_idx).long()

class HFModelWrapper(Module):
    def __init__(self, hf_model, pad_idx):
        self.hf_model, self.pad_idx, self._attention_mask = hf_model, pad_idx, None
    def forward(self, input_ids):
        mask = self._attention_mask if self._attention_mask is not None else (input_ids != self.pad_idx).long()
        self._attention_mask = None
        return self.hf_model(input_ids=input_ids, attention_mask=mask).logits
    def __getitem__(self, idx):
        return self.hf_model.base_model if idx == 0 else self.hf_model.classifier

def get_local_path(folder_name='data'):
    path = Path(folder_name)
    path.mkdir(parents=True, exist_ok=True)
    return path
