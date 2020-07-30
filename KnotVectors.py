import numpy as np
import matplotlib.pyplot as plt

def OpenUniformKnotVector(Order, Points, Normalize):
    nPlusOne = np.size(Points, 0)
    n = nPlusOne - 1
    countKnots = nPlusOne + Order
    knots = np.zeros(countKnots)
    for i in range(countKnots):
        if 0 <= i and i < Order:
            knots[i] = 0.
        elif Order <= i and i <= n:
            knots[i] = i + 1 - Order
        else:
            knots[i] = n - Order + 2

    if Normalize:
        knots /= np.max(knots)
    return knots

def PeriodicKnotVector(Order, CountPoints, Normalize):
    values = np.arange(0., CountPoints + Order, 1.)
    if Normalize:
        values /= np.max(values)
    return values