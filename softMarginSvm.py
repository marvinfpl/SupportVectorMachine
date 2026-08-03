import numpy as np
from cvxopt import matrix, solvers

class SoftMarginSvmClassifier():
    def __init__(self, C: float):
        self.C = C
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray)->None:
        n_samples, n_features = X.shape
        y = y.reshape(1, -1).astype(float)
        
        Q = np.outer(y, y) * X @ X.T
        q = -np.eye(n_samples)
        G = np.eye(n_samples)
        h = -np.eye(n_samples)
        A = y.reshape(1, -1)
        b = [0.0]

        sol = solvers.qp(matrix(Q), matrix(q), matrix(G), matrix(h*self.C), matrix(A), matrix(b))

        alpha = np.array(sol['x']).flatten()

        support = alpha > 1e-6
        weights = X.T @ (alpha * y)
        bias = np.mean(y[support] - X.T[support] @ weights)

        self.weights = weights
        self.bias = bias


    def predict(self, X: np.ndarray)->np.ndarray:
        return np.sign(X.T @ self.weights + self.bias)