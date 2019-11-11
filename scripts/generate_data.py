#!/usr/bin/env python3
"""
This script generate `.csv` file with data for training linear
regression model for equation:

    y = 3x + 2

with the domain of X=[0, 20]
The data will be altered with noise [-1,1]
"""

import csv
from collections import defaultdict
from random import random, randint


def noised_f(x: float) -> float:
    """
    This function calculates value of our function and applies noise 
    on it.
    """
    true_y = 3*x + 2
    noise = random()
    y = true_y + noise if randint(0, 1) else true_y - noise
    return y


def generate_data(entries: int = 50) -> tuple:
    data_x = []
    data_y = []

    for _ in range(entries):
        x = random() * 20
        y = noised_f(x)
        data_x.append(x)
        data_y.append(y)

    return (data_x, data_y)


def save_to_csv(data_x: list, data_y: list) -> None:
    with open("./data.csv", "w") as f:
        writer = csv.writer(f)
        for entry in zip(data_x, data_y):
            writer.writerow(entry)


if __name__ == "__main__":
    data_x, data_y = generate_data()
    save_to_csv(data_x, data_y)
