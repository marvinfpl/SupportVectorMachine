import numpy as np
from cvxopt import matrix
from cvxopt import solvers

class SVMClassifier():
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):

        "Solve an optimisation problem of the form min w,b 1/2 * w * w such that y(w*x + b) >= 1"

        n_samples, n_features = X.size
        w = np.ndarray((1, n_features), dtype=np.dtype(X))
        b = np.ndarray((1, n_features), dtype=np.dtype(X))
        ones = np.ones_like(b, dtype=np.dtype(X))
        zeros = np.ones_like(w, dtype=np.dtype(X))
        q = -np.identity(n_features, dtype=np.dtype(X))
        n_rows, n_labels = y.size
        b_prime = y.T @ b - ones
        g = np.vstack((-np.eye))
        assert n_labels == 1 and n_rows == n_samples

        x_opt = matrix(X)
        y_opt = matrix(y)
        h_opt = matrix(b_prime)
        q_opt = matrix(q)
        zeros_opt = matrix(zeros)

        sol = solvers.qp(q_opt, zeros_opt, x_opt, h_opt)
        w = np.array(sol['x'])
        b = np.mean(y - w.T @ X)





