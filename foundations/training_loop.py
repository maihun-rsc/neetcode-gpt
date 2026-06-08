import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        n = len(X)
        w, b = np.zeros(X.shape[1]), 0
        for _ in range(epochs):
            y_cap = X @ w + b 
            w = w - lr*(2/n)*(X.T @ (y_cap - y))
            b = b - lr*(2/n)*np.sum(y_cap - y)
        return (np.round(w, 5), round(b, 5))
