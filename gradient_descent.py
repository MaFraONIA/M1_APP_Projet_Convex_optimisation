import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def logistic_regression_gd(X, y, lr=0.01, lambda_=0.1, max_iter=1000, tol=1e-6):
    try:
        n, p = X.shape
        w = np.zeros(p)
        for iteration in range(max_iter):
            model = 1 / (1 + np.exp(-X.dot(w)))
            gradient = X.T.dot(model - y) / n + lambda_ * w
            w_old = w.copy()
            w -= lr * gradient
            if np.linalg.norm(w - w_old, ord=2) < tol:
                logging.info(f"Convergence achieved after {iteration} iterations")
                break
        return w
    except Exception as e:
        logging.error("An error occurred in logistic_regression_gd function", exc_info=True)
        return None