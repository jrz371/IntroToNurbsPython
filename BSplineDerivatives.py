import numpy as np
import matplotlib.pyplot as plt
import math
from BSplineBasis import CoxDeBoorRecursion, BasisGraph
from KnotVectors import OpenUniformKnotVector

def CoxDeBoorFirstDerivative(i, order, Knots, t):
    if order == 1:
        return 0.
    else:
        ftt = CoxDeBoorRecursion(i, order - 1, Knots, t) + (t - Knots[i]) * CoxDeBoorFirstDerivative(i, order - 1, Knots, t)
        ftb = Knots[i + order - 1] - Knots[i]
        stt = (Knots[i + order] - t) * CoxDeBoorFirstDerivative(i + 1, order - 1, Knots, t) - CoxDeBoorRecursion(i + 1, order - 1, Knots, t)
        stb = Knots[i + order] - Knots[i + 1]

        first = ftt / ftb
        if math.isnan(first):
            first = 0.

        second = stt / stb
        if math.isnan(second):
            second = 0.

        return first + second

def CoxDeBoorSecondDerivative(i, order, Knots, t):
    if order < 2:
        return 0.
    else:
        ftt = 2 * CoxDeBoorFirstDerivative(i, order - 1, Knots, t) + (t - Knots[i]) * CoxDeBoorSecondDerivative(i, order - 1, Knots, t)
        ftb = Knots[i + order - 1] - Knots[i]
        stt = (Knots[i + order] - t) * CoxDeBoorSecondDerivative(i + 1, order - 1, Knots, t) - 2 * CoxDeBoorFirstDerivative(i + 1, order - 1, Knots, t)
        stb = Knots[i + order] - Knots[i + 1]

        first = ftt / ftb
        if math.isnan(first):
            first = 0.

        second = stt / stb
        if math.isnan(second):
            second = 0.

        return first + second

def FirstDerivativeGraph(Knots, order, i, T):
    values = []
    for t in T:
        values.append(CoxDeBoorFirstDerivative(i, order, Knots, t))
    return np.asarray(values)

def SecondDerivativeGraph(Knots, order, i, T):
    values = []
    for t in T:
        values.append(CoxDeBoorSecondDerivative(i, order, Knots, t))
    return np.asarray(values)

if __name__ == "__main__":
    order = 4
    countPoints = 6
    Knots = OpenUniformKnotVector(order, countPoints, False)

    maxKnot = np.max(Knots)

    T = np.arange(0., maxKnot, 0.01)

    for i in range(countPoints):
        plt.plot(T, BasisGraph(Knots, order, i, T))

    plt.show()

    for i in range(countPoints):
        plt.plot(T, FirstDerivativeGraph(Knots, order, i, T))

    plt.show()

    for i in range(countPoints):
        plt.plot(T, SecondDerivativeGraph(Knots, order, i, T))

    plt.show()