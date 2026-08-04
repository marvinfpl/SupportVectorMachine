import numpy as np
from cvxopt import matrix, solvers
from typing import Literal

class KernelSoftMarginSvmClassifier():
    """
    C: tradeoff between margin and error
    r: value in polynomial kernel : (sigma * a @ b + r)**d and sigmoid kernel : tanh(sigma * a @ b + r)
    sigma: value in some kernels to treat idk what
    d: order of the polynomial
    """
    def __init__(self, C: float=0.95, r: float=1, gamma: float=0.3, d: int=3, kernel: Literal['linear', 'polynomial', 'rbf', 'sigmoid']='linear',):
        self.C = C
        self.r = r
        self.gamma = gamma
        self.d = d
        self.kernel = kernel
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None

    def K(self, a: np.ndarray, b: np.ndarray)->np.ndarray:
        if self.kernel == 'linear':
            return a @ b
        elif self.kernel == 'polynomial':
            return (self.gamma * a @ b + self.r) ** self.d
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(a-b)**2)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * a @ b + self.r)
        else:
            raise Exception("No valid kernel has been selected")

    def gram_matrix(self, a: np.ndarray, b: np.ndarray):
        n1 = a.shape[0]
        n2 = b.shape[0]
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.K(a[i], b[j])
        return K

    def fit(self, X: np.ndarray, y: np.ndarray)->None:
        n_samples, n_features = X.shape
        assert np.all(np.isin(y, [-1, 1]))

        y = y.reshape(-1).astype(float)
        K = self.gram_matrix(X, X)
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        G = np.vstack([
            -np.eye(n_samples),
            np.eye(n_samples),
        ])
        h = np.hstack([
            np.zeros(n_samples),
            self.C*np.ones(n_samples)
        ])
        A = y.reshape(1, -1)
        b = np.array([0.0])

        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
        if sol['status'] != 'optimal':
            raise RuntimeError(sol['status'])
        
        alpha = np.array(sol['x']).flatten()

        support = alpha > 1e-6
        self.alpha = alpha[support]
        self.support_vectors = X[support]
        self.support_labels = y[support]
        Ksv = self.gram_matrix(
            self.support_vectors,
            self.support_vectors
        )
        self.bias = np.mean(self.support_labels - Ksv @ (self.alpha * self.support_labels))

    def predict(self, X: np.ndarray)->np.ndarray:
        K = self.gram_matrix(self.support_vectors, X)
        return np.sign(K.T @ (self.alpha * self.support_labels) + self.bias)