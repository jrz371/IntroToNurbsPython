import numpy as np
import matplotlib.pyplot as plt
import math
from BSplineBasis import CoxDeBoorRecursion
from KnotVectors import OpenUniformKnotVector

def BSpline(Points, Order, Knots, T):
    countPoints = np.size(Points, 0)
    n = countPoints - 1
    dimension = np.size(Points, 1)
    rVal = []
    for t in T:
        point = np.zeros(dimension)
        for i in range(0, n+1):
            point += Points[i] * CoxDeBoorRecursion(i, Order, Knots, t)
        rVal.append(point)
    return np.array(rVal)

if __name__ == "__main__":
    Points = np.array([[1., 1.], [2., 6.], [4., 3.], [6., 6.], [8., 6.]])
    #KnotsA = np.array([0, 0, 0, 0.33, 0.66, 1, 1, 1])
    KnotsA = OpenUniformKnotVector(3, Points, True)
    KnotsB = np.array([0, 0, 0, 0.33, 0.33, 1, 1, 1])
    T = np.arange(0.01, 1.0, 0.01)

    PointsBox = np.array([[0, 0], [2, 0], [4, 0], [4, 2], [4, 4], [2, 4], [0, 4], [0, 2], [0, 0], [2, 0], [4, 0]])
    KnotsBox = np.arange(0, 15, 1)
    BoxT = np.arange(3, 11, 0.01)

    RBox = BSpline(PointsBox, 4, KnotsBox, BoxT)

    RA = BSpline(Points, 3, KnotsA, T)
    RB = BSpline(Points, 3, KnotsB, T)

    plt.plot(Points[:, 0], Points[:, 1])
    plt.plot(RA[:, 0], RA[:, 1])
    plt.plot(RB[:, 0], RB[:, 1])

    #plt.plot(PointsBox[:, 0], PointsBox[:, 1])
    plt.plot(RBox[:, 0], RBox[:, 1])
    plt.show()