import numpy as np
import matplotlib.pyplot as plt
import math
from BSplineBasis import CoxDeBoorRecursion
from KnotVectors import OpenUniformKnotVector

def NurbsBasis(i, order, Knots, t, Weights, countPoints):
    top = Weights[i] * CoxDeBoorRecursion(i, order, Knots, t)

    bottom = 0.
    for j in range(countPoints):
        bottom += Weights[j] * CoxDeBoorRecursion(j, order, Knots, t)

    return top / bottom

def NurbsBasisGraph(Knots, order, i, T, Weights, countPoints):
    values = []
    for t in T:
        values.append(NurbsBasis(i, order, Knots, t, Weights, countPoints))
    return np.asarray(values)

if __name__ == "__main__":
    countPoints = 5
    order = 3
    Knots = OpenUniformKnotVector(order, countPoints, True)
    WeightsA = np.asarray([1, 1, 0, 1, 1])
    WeightsB = np.asarray([1, 1, 0.25, 1, 1])
    WeightsC = np.asarray([1, 1, 1, 1, 1])
    WeightsD = np.asarray([1, 1, 5, 1, 1])

    T = np.arange(0., 1., 0.01)

    for i in range(countPoints):
        plt.plot(T, NurbsBasisGraph(Knots, order, i, T, WeightsA, countPoints))

    plt.show()

    for i in range(countPoints):
        plt.plot(T, NurbsBasisGraph(Knots, order, i, T, WeightsB, countPoints))

    plt.show()

    for i in range(countPoints):
        plt.plot(T, NurbsBasisGraph(Knots, order, i, T, WeightsC, countPoints))

    plt.show()

    for i in range(countPoints):
        plt.plot(T, NurbsBasisGraph(Knots, order, i, T, WeightsD, countPoints))

    plt.show()
