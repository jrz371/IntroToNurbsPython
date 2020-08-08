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

# 0.0 <= T < 1.0, evaluated for each segment
def PeriodicBSplineMatrix(Points, order, T, closed=False):
    TMat = TMatrix(T, order)
    Basis = BasisMatrix(order)
    countPoints = np.size(Points, 0)
    dimension = np.size(Points, 1)

    orderSubTwo = order - 2 #k - 1 repeated/looped, subtract 2 since first vertex already there

    totalPoints = countPoints + orderSubTwo * 2

    PointsMatrix = np.zeros([totalPoints, dimension])
    PointsMatrix[orderSubTwo:orderSubTwo + countPoints, :] = Points

    if closed:
        PointsMatrix[0:orderSubTwo] = Points[-orderSubTwo:, :]
        PointsMatrix[orderSubTwo + countPoints:, :] = Points[:orderSubTwo, :]
    else:
        PointsMatrix[0:orderSubTwo] = Points[0, :]
        PointsMatrix[orderSubTwo + countPoints:, :] = Points[-1, :]

    RFinal = np.empty([0, dimension])

    #have to iterate over each segment of the curve
    for i in range(0, totalPoints - order + 1):
        RSegment = np.dot(np.dot(TMat, Basis), PointsMatrix[i:i+order, :])
        RFinal = np.append(RFinal, RSegment, axis=0)
    
    return RFinal

if __name__ == "__main__":
    order = 3

    Points = np.array([[0., 0.], [3., 10.], [6., 3.], [10., 5.]])
    countPoints = np.size(Points, 0)

    T = np.arange(0.0, 1.0, 0.01)

    Spline = PeriodicBSplineMatrix(Points, order, T, False)

    plt.plot(Points[:, 0], Points[:, 1])
    plt.plot(Spline[:, 0], Spline[:, 1])

    PointsBox = np.array([[2, 0], [4, 0], [4, 2], [4, 4], [2, 4], [0, 4], [0, 2], [0, 0]])

    BoxSpline = PeriodicBSplineMatrix(PointsBox, order, T, True)
    plt.plot(BoxSpline[:, 0], BoxSpline[:, 1])

    plt.show()