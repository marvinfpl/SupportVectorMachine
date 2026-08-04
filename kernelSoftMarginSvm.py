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

    def K(self, a: np.ndarray, b: np.ndarray)->np.ndarray:
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
        assert np.all(np.isin(y, [-1, 1]))

        y = y.reshape(-1).astype(float)

        P = np.outer(y, y) * self.K(X, X)
        q = -np.eye(n_samples)
        G = np.vstack([
            -np.eye(n_samples),
            np.eye(n_samples),
        ])
        h = np.hstack([
            np.zeros(n_samples),
            -self.C*np.ones(n_samples)
        ])
        A = y.reshape(1, -1)
        b = np.array([0.0])

        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
        if sol['status'] != 'optimal':
            raise RuntimeError(sol['status'])
        
        alpha = np.array(sol['x']).flatten()

        support = alpha > 1e-6
        weights = alpha * y
        bias = np.mean(y[support] - self.K(X, X).T @ (alpha[support] * y[support]))
        self.weights = weights
        self.bias = bias

    def predict(self, X: np.ndarray)->np.ndarray:
        return np.sign(self.K(X, X).T @ self.weights + self.bias)