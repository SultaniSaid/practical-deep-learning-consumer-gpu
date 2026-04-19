try:
    import torch_directml
    print(torch_directml.device())
    import torch

    d = torch_directml.device()
    x = torch.ones((2, 2), device=d, dtype=torch.float32)
    print(x.cpu())
except ImportError:
    print("torch_directml not available")