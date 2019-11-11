"""
This module contains python implementation of linear regression algorithm,
that I will be then implement in CUDA. I will be using data generated from
`generate_data.py`, which generates data of function `y = 3x + 2`.

The model will is of form `y = X * W` where `X` is a vector of features and
`W` is a vector of weights.
"""

import numpy as np

from generate_data import generate_data


def mse(true_y: "np.array", pred_y: "np.array") -> float:
    """
    This function calculates the mean squared error (MSE) of the model
    based on its predictions (`pred_y`) on true labels (`true_y`).
    """
    return np.mean((true_y - pred_y)**2)


def calculate_weights(X, y):
    return np.linalg.inv(X.transpose() @ X) @ X.transpose() @ y


if __name__ == "__main__":
    data_x, data_y = generate_data(1000000)

    X = np.array(data_x).reshape(len(data_x), 1)
    y = np.array(data_y)

    X = np.insert(X, 0, 1, axis=1)

    weights = calculate_weights(X, y)

    pred = X @ weights
    print(f"mse: {mse(y, pred)}")
    print(f"predicted: y = {weights[1]} * x + {weights[0]}")
    print("true: y = 3 * x + 2")
