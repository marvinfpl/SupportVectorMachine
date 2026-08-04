import numpy as np
from cvxopt import matrix, solvers

class KernelSoftMarginSvmClassifier():
    """
    C: tradeoff between margin and error
    r: value in polynomial kernel : (sigma * a @ b + r)**d and sigmoid kernel : tanh(sigma * a @ b + r)
    sigma: value in some kernels to treat idk what
    d: order of the polynomial
    """
    def __init__(self, C: float=0.95, r: float=1, sigma: float=0.3, d: int=3, kernel='linear'|'polynomial'|'rbf'|'sigmoid',):
        self.C = C
        self.r = r
        self.sigma = sigma
        self.d = d
        self.kernel = kernel
        self.weights = None
        self.bias = None

    def kernel(self, a: np.ndarray, b: np.ndarray)->np.ndarray:
        a_rows, a_cols = a.shape
        b_rows, b_cols = b.shape
        if a_rows == b_cols and a_cols == b_rows:
            pass
        elif a_rows == b_rows and a_cols == b_cols:
            a = a.T
        else:
            raise Exception("A and B are not eligible for dot product")

        if self.kernel == 'linear':
            return a @ b
        elif self.kernel == 'polynomial':
            return (self.sigma * a @ b + self.r) ** self.d
        elif self.kernel == 'rbf':
            return np.exp(-self.sigma * np.linalg.norm(a-b, 2))
        elif self.kernel == 'sigmoid':
            return np.tanh(self.sigma * a @ b + self.r)
        else:
            raise Exception("No valid kernel has been selected")

    def fit(self, X: np.ndarray, y: np.ndarray)->None:
        n_samples, n_features = X.shape
        

    def predict(self, X: np.ndarray):
        pass