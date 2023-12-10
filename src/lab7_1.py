import numpy as np, random
import matplotlib.pyplot as plt

lr = 1
bias = 1
weights = list()
for k in range(3):
    weights.append(random.random())  # Assigning random weights


def sig(x: float) -> float:
    return 1.0 / (1 + (np.exp(-x)))


def ptron(x1, x2, y):
    out_value = go_forward(x1, x2)
    err = y - out_value
    weights[0] += err * x1 * lr  # Modifying weights
    weights[1] += err * x2 * lr
    weights[2] += err * bias * lr


def go_forward(x1, x2):
    return sig(x1 * weights[0] + x2 * weights[1] + bias * weights[2])


def f(x, k, b):
    return k * x + b


def predict(value):
    if value >= 0.5:
        return 1
    else:
        return 0


if __name__ == '__main__':
    for i in range(50):  # Training With Data
        ptron(0, 0, 0)  # Passing the tryth values of OR
        ptron(1, 1, 1)
        ptron(1, 0, 1)
        ptron(0, 1, 1)

    Xs = [0, 1, 0, 1]
    Ys = [0, 0, 1, 1]
    for x, y in zip(Xs, Ys):
        out = go_forward(x, y)
        print(x, "OR", y, "=>:", predict(out))

    x = np.linspace(0, 1, 100)
    y = []

    plt.scatter(Xs[1:], Ys[1:], c="blue")
    plt.scatter(Xs[:1], Ys[:1], c="red")

    k = -(weights[0] / weights[1])
    b = -(weights[2] / weights[1])
    for x_i in x:
        y.append(f(x_i, k, b))
    plt.plot(x, y)
    plt.show()
