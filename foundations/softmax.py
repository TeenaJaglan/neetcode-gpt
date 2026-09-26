import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxi = max(z)
        t = np.ndarray(len(z))
        j = 0
        sum = 0;
        for i in z:
            sum += np.exp(i-maxi)
        for i in z:
            n = np.exp(i-maxi)
            t[j] = np.round(n/sum,4)
            j=j+1
        return t;
