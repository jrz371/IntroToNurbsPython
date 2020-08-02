import numpy as np
import matplotlib.pyplot as plt
from Bezier import Combination
from KnotVectors import PeriodicKnotVector
from BSpline import BSpline

def TMatrix(T, order):
    countT = np.size(T)
    TMat = np.ones([countT, order])

    for i in range(order - 1):
        TMat[:, i] = np.power(T, order - i - 1)
    
    return TMat

def BasisSum(i, j, order):
    value = 0.
    for l in range(j, order):
        value += np.power((order - (l + 1)), i) * np.power(-1., l - j) * Combination(order, l - j)
    return value

def BasisMatrix(order):
    Mat = np.zeros([order, order])

    for i in range(order):
        for j in range(order):
            ki = (1.0 / np.math.factorial(order - 1)) * Combination(order - 1, i)
            Mat[i, j] = ki * BasisSum(i, j , order)
    return Mat

if __name__ == "__main__":
    order = 4

    Points = np.array([[0., 0.], [3., 10.], [6., 3.], [10., 5.]])
    countPoints = np.size(Points, 0)

    T = np.arange(0.0, 1.0, 0.01)

    TMat = TMatrix(T, order)

    Basis = BasisMatrix(order)

    plt.plot(Points[:, 0], Points[:, 1])

    #have to iterate over each segment of the curve
    for i in range(0, countPoints - order + 1):
        RSegment = np.dot(np.dot(TMat, Basis), Points[i:i+order, :])
        plt.plot(RSegment[:, 0], RSegment[:, 1])

    plt.show()