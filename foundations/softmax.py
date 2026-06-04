import numpy as np
from numpy.typing import NDArray
import math as muth

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        answers, denom = np.array([(np.round(muth.e**(x - max(z)), 4)) for x in z]), 0
        for i in z:
            denom += (muth.e**(i - max(z)))
        for i in range(len(answers)):
            answers[i] = np.round(answers[i]/denom, 4)
        
        return answers
            
