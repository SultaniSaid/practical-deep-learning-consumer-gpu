import importlib.util
import traceback

print('python executable:', __import__('sys').executable)
print('torch installed:', importlib.util.find_spec('torch') is not None)
print('torch_directml installed:', importlib.util.find_spec('torch_directml') is not None)

try:
    import torch
    print('torch_version:', torch.__version__)
    print('cuda_available:', torch.cuda.is_available() if hasattr(torch, 'cuda') else False)
except Exception:
    traceback.print_exc()

try:
    import torch_directml
    print('torch_directml module:', torch_directml)
    try:
        device = torch_directml.device()
        print('device():', device)
        x = torch.ones((2, 2), device=device)
        print('tensor_device:', x.device)
    except Exception:
        traceback.print_exc()
except Exception:
    traceback.print_exc()
