import numpy as np
from numpy.typing import NDArray
import torch


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        mean = 0
        n = len(x)
        for i in range(n):
            mean += x[i]/n
        sd = x - mean
        var = sum(sd**2)/n
        denom = (var+10**-5)**0.5
        x_hat = []
        for i in range(n):
            dev = x[i] - mean
            xi_hat = (dev/denom)*gamma[i] + beta[i]
            x_hat.append(round(xi_hat, 5))
        return x_hat

