import numpy as np
from cvxopt import matrix
from cvxopt import solvers

class SVMClassifier():
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray)->None:

        "Solve an optimisation problem of the form min w,b 1/2 * w * w such that y(w*x + b) >= 1"

        n_samples, n_features = X.shape
        y = y.reshape(1, -1).astype(float)
        Q = np.outer(y, y) * X @ X.T
        q = -np.ones(n_samples)
        G = -np.eye(n_samples)
        h = np.zeros(n_samples)
        A = y
        b = np.zeros(n_samples)

        sol = solvers.qp(matrix(Q), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
        alpha = np.array(sol['x']).flatten()

        weights = X.T @ alpha @ y
        bias = np.mean(y -  X @ weights)
        self.weights = weights
        self.bias = bias

    def predict(self, X: np.ndarray)->np.ndarray:
        results = np.sign(X @ self.weights + self.bias)
        results[results > 0] = 1
        results[results < 0] = -1
        return results