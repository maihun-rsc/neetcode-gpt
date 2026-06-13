import torch
import torch.nn as nn
from torchtyping import TensorType
from math import sqrt

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.attention_dim = attention_dim
        self.key = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias = False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        K, Q, V = self.key(embedded), self.query(embedded), self.value(embedded)
        attention_scores = Q @ K.transpose(-2, -1) / ((self.attention_dim)**0.5)
 
        lower_triangular = torch.tril(torch.ones(K.shape[1], K.shape[1])) 
        mask = lower_triangular == 0
        masked_scores = attention_scores.masked_fill(mask, float('-inf'))
        masked_scores = nn.functional.softmax(masked_scores, dim=2)
        
        return torch.round(masked_scores @ V, decimals = 4)
