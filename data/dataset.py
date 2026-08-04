import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: 
        torch.manual_seed(0)
        tokens = raw_dataset.split()
        # 2. Generate batch_size random start indices using 
        batches = torch.randint(low = 0, high = len(tokens) - context_length, size=(batch_size,)).tolist()
        X, Y = [], []
        #    Range: [0, len(tokens) - context_length)
        for i in batches:
            X.append(tokens[i:i+context_length])
            Y.append(tokens[i+1:i+1+context_length])
        return X, Y
        pass
