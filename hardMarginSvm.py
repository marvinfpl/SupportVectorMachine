import numpy as np
from cvxopt import matrix
from cvxopt import solvers

class HardMarginSvmClassifier():
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray)->None:

        "Solve an optimisation problem of the form min w,b 1/2 * w * w such that y(w*x + b) >= 1"

        n_samples, n_features = X.shape
        y = y.reshape(-1).astype(float)
        Q = np.outer(y, y) * X @ X.T
        q = -np.ones(n_samples)
        G = -np.eye(n_samples)
        h = np.zeros(n_samples)
        A = y.reshape(1, -1)
        b = [0.0]

        sol = solvers.qp(matrix(Q), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
        alpha = np.array(sol['x']).flatten()

        weights = X.T @ (alpha * y)
        support = alpha > 1e-6
        bias = np.mean(y[support] -  X[support] @ weights)
        self.weights = weights
        self.bias = bias

    def predict(self, X: np.ndarray)->np.ndarray:
        return np.sign(X @ self.weights + self.bias)