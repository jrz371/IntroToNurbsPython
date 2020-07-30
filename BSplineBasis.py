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

def BasisGraph(Knots, Order, i, T):
    values = []
    for t in T:
        values.append(CoxDeBoorRecursion(i, Order, Knots, t))
    return np.asarray(values)


if __name__ == "__main__":
    Order = 3
    CountPoints = 4
    Knots = OpenUniformKnotVector(Order, np.zeros([CountPoints, 2]), True)

    T = np.arange(0., 1.0, 0.01)

    for i in range(CountPoints):
        plt.plot(BasisGraph(Knots, Order, i, T))
    
    plt.show()

    Knots = PeriodicKnotVector(Order, CountPoints, True)

    for i in range(CountPoints):
        plt.plot(BasisGraph(Knots, Order, i, T))
    
    plt.show()

