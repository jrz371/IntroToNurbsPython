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

# 0.0 <= T < 1.0
def PeriodicBSplineMatrix(Points, order, T):
    TMat = TMatrix(T, order)
    Basis = BasisMatrix(order)
    dimension = np.size(Points, 1)

    RFinal = np.empty([0, dimension])

    #have to iterate over each segment of the curve
    for i in range(0, countPoints - order + 1):
        RSegment = np.dot(np.dot(TMat, Basis), Points[i:i+order, :])
        RFinal = np.append(RFinal, RSegment, axis=0)
    
    return RFinal

if __name__ == "__main__":
    order = 3

    Points = np.array([[0., 0.], [3., 10.], [6., 3.], [10., 5.]])
    countPoints = np.size(Points, 0)

    T = np.arange(0.0, 1.0, 0.01)

    R = PeriodicBSplineMatrix(Points, order, T)

    plt.plot(Points[:, 0], Points[:, 1])
    plt.plot(R[:, 0], R[:, 1])

    #usable parameter range 2 <= T <= 4 when order = 3
    TConfirm = np.arange(2, 4, 0.01)
    RConfirm = BSpline(Points + [1, 1], order, PeriodicKnotVector(order, countPoints, False), TConfirm)
    plt.plot(RConfirm[:, 0], RConfirm[:, 1])

    plt.show()