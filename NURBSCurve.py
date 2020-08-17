import numpy as np
import matplotlib.pyplot as plt
import math
from NurbsBasis import NurbsBasis
from KnotVectors import OpenUniformKnotVector

def NurbsCurve(Points, order, Knots, Weights, T):
    countPoints = np.size(Points, 0)
    dimension = np.size(Points, 1)
    Return = []
    for t in T:
        val = np.zeros(dimension)
        for i in range(countPoints):
            val += Points[i, :] * NurbsBasis(i, order, Knots, t, Weights, countPoints)
        Return.append(val)
    return np.asarray(Return)

if __name__ == "__main__":
    Points = np.asarray([[0, 0], [1, 2], [2.5, 0], [4, 2], [5, 0]])
    countPoints = np.size(Points, 0)
    order = 3
    Knots = OpenUniformKnotVector(order, countPoints, True)
    WeightsA = np.asarray([1, 1, 0, 1, 1])
    WeightsB = np.asarray([1, 1, 0.25, 1, 1])
    WeightsC = np.asarray([1, 1, 1, 1, 1])
    WeightsD = np.asarray([1, 1, 5, 1, 1])

    T = np.arange(0., 1., 0.01)

    NurbsCurveA = NurbsCurve(Points, order, Knots, WeightsA, T)
    NurbsCurveB = NurbsCurve(Points, order, Knots, WeightsB, T)
    NurbsCurveC = NurbsCurve(Points, order, Knots, WeightsC, T)
    NurbsCurveD = NurbsCurve(Points, order, Knots, WeightsD, T)

    plt.plot(Points[:, 0], Points[:, 1])
    plt.plot(NurbsCurveA[:, 0], NurbsCurveA[:, 1])
    plt.plot(NurbsCurveB[:, 0], NurbsCurveB[:, 1])
    plt.plot(NurbsCurveC[:, 0], NurbsCurveC[:, 1])
    plt.plot(NurbsCurveD[:, 0], NurbsCurveD[:, 1])
    plt.show()