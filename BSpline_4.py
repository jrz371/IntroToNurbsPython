import numpy as np
import matplotlib.pyplot as plt
import math

def CoxDeBoorRecursion(i, order, Knots, t):
    if order == 1:
        return Knots[i] <= t and t < Knots[i + 1]
    else:
        ftt = (t - Knots[i]) * CoxDeBoorRecursion(i, order - 1, Knots, t)
        ftb = Knots[i + order - 1] - Knots[i]
        stt = (Knots[i + order] - t) * CoxDeBoorRecursion(i + 1, order - 1, Knots, t)
        stb = Knots[i + order] - Knots[i + 1]

        first = ftt / ftb
        if math.isnan(first):
            first = 0.

        second = stt / stb
        if math.isnan(second):
            second = 0.

        return first + second


def BSpline(Points, order, Knots, T):
    countPoints = np.size(Points, 0)
    n = countPoints - 1
    dimension = np.size(Points, 1)
    degree = order - 1
    rVal = []
    for t in T:
        point = np.zeros(dimension)
        for i in range(1, n + 1):
            point += Points[i] * CoxDeBoorRecursion(i, order, Knots, t)
        rVal.append(point)
    return np.array(rVal)

if __name__ == "__main__":
    Points = np.array([[0., 0.], [1., -1.], [2., 0.], [3., -1.]])
    Knots = np.array([0, 0, 0, 0.5, 1, 1, 1, 1])
    T = np.arange(0.01, 1.0, 0.01)

    R = BSpline(Points, 3, Knots, T)

    plt.plot(Points[:, 0], Points[:, 1])
    plt.plot(R[:, 0], R[:, 1])
    plt.show()