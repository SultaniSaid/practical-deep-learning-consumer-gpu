import torch
import torch_directml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = torch_directml.device()

# dummy dataset
xb = torch.randn(100, 28*28)
yb = torch.randint(0, 10, (100,))
dataset = TensorDataset(xb, yb)
dataloader = DataLoader(dataset, batch_size=16)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for xb, yb in dataloader:
    xb = xb.to(device, dtype=torch.float32)
    yb = yb.to(device, dtype=torch.long)

    optimizer.zero_grad()
    out = model(xb)
    loss = loss_fn(out, yb)
    loss.backward()
    optimizer.step()

    print(loss.item())