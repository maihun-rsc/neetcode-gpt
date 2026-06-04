import numpy as np
from numpy.typing import NDArray
import math

class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        anslist = []
        for i in z:
            answer = 1/(1 + math.e**(-i))
            anslist.append(np.round(answer, 5))
        return np.array(anslist)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        anslist = []
        for i in z:
            anslist.append(np.round(max(0.0, i), 2))
        return np.array(anslist)
