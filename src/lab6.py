import numpy as np


def sig(x: float) -> float:
    return 1 / (1 + (np.exp(-x)))


def tanh(x: float) -> float:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


if __name__ == '__main__':
    print(f"Sig: {sig(0.705)}")
    print(f"Tanh: {sig(0.705)}")
