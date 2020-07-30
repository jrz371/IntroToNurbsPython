import numpy as np
import matplotlib.pyplot as plt

def OpenUniformKnotVector(order, countPoints, normalize):
    n = countPoints - 1
    countKnots = countPoints + order
    Knots = np.zeros(countKnots)
    for i in range(countKnots):
        if 0 <= i and i < order:
            Knots[i] = 0.
        elif order <= i and i <= n:
            Knots[i] = i + 1 - order
        else:
            Knots[i] = n - order + 2

    if normalize:
        Knots /= np.max(Knots)
    return Knots

def PeriodicKnotVector(order, countPoints, normalize):
    Knots = np.arange(0., countPoints + order, 1.)
    if normalize:
        Knots /= np.max(Knots)
    return Knots