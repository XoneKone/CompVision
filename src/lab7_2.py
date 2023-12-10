import numpy as np
import matplotlib.pyplot as plt
import random


def run():
    # A utility function to calculate area
    # of triangle formed by (x1, y1),
    # (x2, y2) and (x3, y3)
    def area(x1, y1, x2, y2, x3, y3):

        return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                    + x3 * (y1 - y2)) / 2.0)

    # A function to check whether point P(x, y)
    # lies inside the triangle formed by
    # A(x1, y1), B(x2, y2) and C(x3, y3)
    def isInside(x, y, x1=3, y1=3, x2=6, y2=6, x3=9, y3=1):

        # Calculate area of triangle ABC
        A = area(x1, y1, x2, y2, x3, y3)

        # Calculate area of triangle PBC
        A1 = area(x, y, x2, y2, x3, y3)

        # Calculate area of triangle PAC
        A2 = area(x1, y1, x, y, x3, y3)

        # Calculate area of triangle PAB
        A3 = area(x1, y1, x2, y2, x, y)

        # Check if sum of A1, A2 and A3
        # is same as A
        if A != (A1 + A2 + A3):
            return False
        return True

    def generate_marked_dots(n=1000):
        count = 0
        count_true = 0
        in_triangle = n * 80 // 100
        dots = []
        while count_true <= in_triangle and count <= n:
            x = random.randint(3, 9)
            y = random.randint(1, 6)
            if isInside(x, y):
                dots.append([x, y, 1])
                count_true += 1
            else:
                if count <= n:
                    dots.append([x, y, 0])
            count += 1
        return dots

    def act_perceptron(u):
        if u < 0:
            return 0
        if u > 0:
            return 1

    def go_forward_perceptron(input_vectors, W):
        u = np.dot(input_vectors, W)
        y = act_perceptron(u)
        return y

    def clarification_weights_perceptron(y, x, W):
        if y == 0 and x[-1] == 1:
            W = W + x[:2]
        if y == 1 and x[-1] == 0:
            W = W - x[:2]
        return W

    def predict(value):
        return 1 if value >= 0.5 else 0

    def train(epoch, W, go_forward_func, clarification_weights_func):
        count = len(epoch)
        for k in range(count):
            x = epoch[k]
            y = go_forward_func(x[:2], W)
            W = clarification_weights_func(y, x, W)
        return W

    W1 = np.array([round(np.random.uniform(-1, 1), 2),
                   round(np.random.uniform(-1, 1), 2)])

    train_marked_dots = generate_marked_dots(2000)

    W1 = train(train_marked_dots,
               W1,
               go_forward_perceptron,
               clarification_weights_perceptron)

    test_marked_dots = generate_marked_dots(20)

    X = np.array([[3, 3], [6, 6], [9, 1]])
    Y = ['black', 'black', 'black']

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=170, color=Y[:])

    t1 = plt.Polygon(X[:3, :], color=Y[0])
    plt.gca().add_patch(t1)

    true_list_dots = []
    false_list_dots = []
    mistakes = 0
    for x in test_marked_dots:
        y = predict(go_forward_perceptron(x[:2], W1))

        if y != x[-1]:
            mistakes += 1
            false_list_dots.append(x[:2])
        else:
            true_list_dots.append(x[:2])

    dots_in_triangle = [i for i in test_marked_dots if isInside(i[0], i[1])]

    plt.scatter(x=[i[0] for i in test_marked_dots],
                y=[i[1] for i in test_marked_dots],
                c="blue")

    plt.scatter(x=[i[0] for i in dots_in_triangle],
                y=[i[1] for i in dots_in_triangle],
                c="red")

    plt.scatter(x=[i[0] for i in true_list_dots],
                y=[i[1] for i in true_list_dots],
                c="purple",
                marker=7)

    plt.scatter(x=[i[0] for i in false_list_dots],
                y=[i[1] for i in false_list_dots],
                c="yellow",
                marker=5)

    print(f"Всего предсказаний => {len(test_marked_dots)}")
    print(f"Количество верных предсказаний => {len(test_marked_dots) - mistakes}")
    print(f"Количество ошибок => {mistakes}")
    print(f"Процент правильных ответов => {(len(test_marked_dots) - mistakes) / len(test_marked_dots)}")

    plt.show()


if __name__ == '__main__':
    run()
