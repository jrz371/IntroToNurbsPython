import numpy as np
import matplotlib.pyplot as plt
import math
from NURBSBasis import NURBSBasis
from KnotVectors import OpenUniformKnotVector

def NURBSCurve(Points, order, Knots, Weights, T):
    countPoints = np.size(Points, 0)
    dimension = np.size(Points, 1)
    Return = []
    for t in T:
        val = np.zeros(dimension)
        for i in range(countPoints):
            val += Points[i, :] * NURBSBasis(i, order, Knots, t, Weights, countPoints)
        Return.append(val)
    return np.asarray(Return)

if __name__ == "__main__":
    Points = np.asarray([[0, 0], [1, 2], [2.5, 0], [4, 2], [5, 0]])
    CountPoints = np.size(Points, 0)
    Order = 3
    Knots = OpenUniformKnotVector(Order, CountPoints, True)
    WeightsA = np.asarray([1, 1, 0, 1, 1])
    WeightsB = np.asarray([1, 1, 0.25, 1, 1])
    WeightsC = np.asarray([1, 1, 1, 1, 1])
    WeightsD = np.asarray([1, 1, 5, 1, 1])

    T = np.arange(0., 1., 0.01)

    NurbsCurveA = NURBSCurve(Points, Order, Knots, WeightsA, T)
    NurbsCurveB = NURBSCurve(Points, Order, Knots, WeightsB, T)
    NurbsCurveC = NURBSCurve(Points, Order, Knots, WeightsC, T)
    NurbsCurveD = NURBSCurve(Points, Order, Knots, WeightsD, T)

    plt.plot(Points[:, 0], Points[:, 1])
    plt.plot(NurbsCurveA[:, 0], NurbsCurveA[:, 1])
    plt.plot(NurbsCurveB[:, 0], NurbsCurveB[:, 1])
    plt.plot(NurbsCurveC[:, 0], NurbsCurveC[:, 1])
    plt.plot(NurbsCurveD[:, 0], NurbsCurveD[:, 1])
    plt.show()