import numpy as np
import matplotlib.pyplot as plt
import math
from BSplineBasis import CoxDeBoorRecursion
from KnotVectors import OpenUniformKnotVector

def NURBSBasis(i, order, Knots, t, Weights, countPoints):
    top = Weights[i] * CoxDeBoorRecursion(i, order, Knots, t)

    bottom = 0.
    for j in range(countPoints):
        bottom += Weights[j] * CoxDeBoorRecursion(j, order, Knots, t)

    return top / bottom

def NURBSBasisGraph(Knots, order, i, T, Weights, countPoints):
    values = []
    for t in T:
        values.append(NURBSBasis(i, order, Knots, t, Weights, countPoints))
    return np.asarray(values)

if __name__ == "__main__":
    CountPoints = 5
    Order = 3
    Knots = OpenUniformKnotVector(Order, CountPoints, True)
    WeightsA = np.asarray([1, 1, 0, 1, 1])
    WeightsB = np.asarray([1, 1, 0.25, 1, 1])
    WeightsC = np.asarray([1, 1, 1, 1, 1])
    WeightsD = np.asarray([1, 1, 5, 1, 1])

    T = np.arange(0., 1., 0.01)

    for i in range(CountPoints):
        plt.plot(T, NURBSBasisGraph(Knots, Order, i, T, WeightsA, CountPoints))

    plt.show()

    for i in range(CountPoints):
        plt.plot(T, NURBSBasisGraph(Knots, Order, i, T, WeightsB, CountPoints))

    plt.show()

    for i in range(CountPoints):
        plt.plot(T, NURBSBasisGraph(Knots, Order, i, T, WeightsC, CountPoints))

    plt.show()

    for i in range(CountPoints):
        plt.plot(T, NURBSBasisGraph(Knots, Order, i, T, WeightsD, CountPoints))

    plt.show()
