import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.Linear = nn.Linear(784, 512)
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(p = 0.2)
        self.Projection = nn.Linear(512, 10)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        model = self.Linear(images)
        model = self.ReLU(model)
        model = self.Dropout(model)
        model = self.Projection(model)
        model = self.Sigmoid(model)
        return torch.round(model, decimals = 4)
