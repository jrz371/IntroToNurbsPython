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
    rVal = []
    for t in T:
        point = np.zeros(dimension)
        for i in range(1, n + 1):
            point += Points[i] * CoxDeBoorRecursion(i, order, Knots, t)
        rVal.append(point)
    return np.array(rVal)

if __name__ == "__main__":
    Points = np.array([[0., 0.], [2., 6.], [4., 3.], [6., 6.], [8., 6.]])
    KnotsA = np.array([0, 0, 0, 0.33, 0.66, 1, 1, 1])
    KnotsB = np.array([0, 0, 0, 0.33, 0.33, 1, 1, 1])
    T = np.arange(0.01, 1.0, 0.01)

    PointsBox = np.array([[0, 0], [2, 0], [4, 0], [4, 2], [4, 4], [2, 4], [0, 4], [0, 2], [0, 0], [2, 0], [4, 0]])
    KnotsBox = np.arange(0, 15, 1) / 14

    RBox = BSpline(PointsBox, 4, KnotsBox, T)

    RA = BSpline(Points, 3, KnotsA, T)
    RB = BSpline(Points, 3, KnotsB, T)

    plt.plot(Points[:, 0], Points[:, 1])
    plt.plot(RA[:, 0], RA[:, 1])
    plt.plot(RB[:, 0], RB[:, 1])

    #plt.plot(PointsBox[:, 0], PointsBox[:, 1])
    plt.plot(RBox[:, 0], RBox[:, 1])
    plt.show()