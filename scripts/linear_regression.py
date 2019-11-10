"""
This module contains python implementation of linear regression algorithm,
that I will be then implement in CUDA. I will be using data generated from
`generate_data.py`, which generates data of function `y = 3x + 2`.
"""

import numpy as np

from generate_data import generate_data


def mse(true_y: "np.array", pred_y: "np.array") -> float:
    """
    This function calculates the mean squared error (MSE) of the model
    based on its predictions (`pred_y`) on true labels (`true_y`).
    """
    return np.mean((true_y - pred_y)**2)


def calculate_slope(X: "np.array", y: "np.array") -> float:
    """
    This function calculates the slope of the predicted function.
    """
    mean_x = np.mean(X)
    mean_y = np.mean(y)
    up = 0
    down = 0
    for i in range(len(X)):
        up += (X[i] - mean_x) * (y[i] - mean_y)
        down += (X[i] - mean_x)**2

    return up/down


def calculate_intrecept(X: "np.array", y: "np.array", coefficient: float) -> float:
    """
    This function calculates the independent term of the linear function.
    """
    return np.mean(y) - coefficient * np.mean(X)


if __name__ == "__main__":
    X, y = generate_data()

    slope = calculate_slope(X, y)
    intrecept = calculate_intrecept(X, y, slope)

    print(f"y = {slope} * x + {intrecept}")
