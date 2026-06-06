import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        prediction = np.matmul(X, weights)
        return np.round(prediction, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        ans = 0
        for i in range(len(ground_truth)):
            ans += ((model_prediction[i][0] - ground_truth[i][0])**2)/len(ground_truth)
        return round(ans, 5)