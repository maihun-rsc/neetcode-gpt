import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        sum = 0
        for i in range(len(y_true)):
            sum += (-1/len(y_true))*(y_true[i]*math.log(y_pred[i]) + (1 - y_true[i])*math.log(1 - y_pred[i]))
        return round(sum, 4)
    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        sum = 0
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                sum += (-1/len(y_true))*(y_true[i][j]*math.log(y_pred[i][j]))
        return round(sum, 4)