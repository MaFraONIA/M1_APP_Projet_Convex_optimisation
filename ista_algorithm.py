import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ISTA(X, y, lambda_, max_iter=100, tol=1e-6):
    """
    Iterative Soft-Thresholding Algorithm (ISTA) for logistic regression with Lasso penalty.

    Parameters:
    - X: numpy.ndarray, feature matrix.
    - y: numpy.ndarray, target vector.
    - lambda_: float, regularization parameter.
    - max_iter: int, maximum number of iterations.
    - tol: float, tolerance for convergence.

    Returns:
    - w: numpy.ndarray, the final weight vector.
    """
    try:
        n, p = X.shape
        w = np.zeros(p)
        for _ in range(max_iter):
            w_old = w.copy()
            grad = X.T @ (X @ w - y) / n  # Compute gradient
            # Apply soft-thresholding
            w = np.sign(w - grad) * np.maximum(np.abs(w - grad) - lambda_, 0)
            # Check for convergence
            if np.linalg.norm(w - w_old) < tol:
                logging.info("ISTA algorithm converged after {} iterations.".format(_ + 1))
                break
        return w
    except Exception as e:
        logging.error("Error in ISTA algorithm: %s", e, exc_info=True)
        raise