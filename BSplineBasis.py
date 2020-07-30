import numpy as np
import matplotlib.pyplot as plt
import math
from KnotVectors import OpenUniformKnotVector, PeriodicKnotVector

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

def BasisGraph(Knots, order, i, T):
    values = []
    for t in T:
        values.append(CoxDeBoorRecursion(i, order, Knots, t))
    return np.asarray(values)


if __name__ == "__main__":
    order = 3
    countPoints = 4
    Knots = OpenUniformKnotVector(order, countPoints, True)

    T = np.arange(0., 1.0, 0.01)

    for i in range(countPoints):
        plt.plot(BasisGraph(Knots, order, i, T))
    
    plt.show()

    Knots = PeriodicKnotVector(order, countPoints, True)

    for i in range(countPoints):
        plt.plot(BasisGraph(Knots, order, i, T))
    
    plt.show()

