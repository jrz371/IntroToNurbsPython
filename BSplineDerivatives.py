import numpy as np
import matplotlib.pyplot as plt
import math
from BSplineBasis import CoxDeBoorRecursion, BasisGraph
from KnotVectors import OpenUniformKnotVector
from BSpline import BSpline

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
    if order <= 2:
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

def BasisFirstDerivativeGraph(Knots, order, i, T):
    values = []
    for t in T:
        values.append(CoxDeBoorFirstDerivative(i, order, Knots, t))
    return np.asarray(values)

def BasisSecondDerivativeGraph(Knots, order, i, T):
    values = []
    for t in T:
        values.append(CoxDeBoorSecondDerivative(i, order, Knots, t))
    return np.asarray(values)

def BSplineFirstDerivative(Points, order, Knots, T):
    countPoints = np.size(Points, 0)
    n = countPoints - 1
    dimension = np.size(Points, 1)
    rVal = []
    for t in T:
        point = np.zeros(dimension)
        for i in range(0, n+1):
            point += Points[i] * CoxDeBoorFirstDerivative(i, order, Knots, t)
        rVal.append(point)
    return np.array(rVal)

def BSplineSecondDerivative(Points, order, Knots, T):
    countPoints = np.size(Points, 0)
    n = countPoints - 1
    dimension = np.size(Points, 1)
    rVal = []
    for t in T:
        point = np.zeros(dimension)
        for i in range(0, n+1):
            point += Points[i] * CoxDeBoorSecondDerivative(i, order, Knots, t)
        rVal.append(point)
    return np.array(rVal)

def BSplineTangent(Points, order, Knots, t):
    deriv = BSplineFirstDerivative(Points, order, Knots, [t])[0]
    point = BSpline(Points, order, Knots, [t])[0]
    norm = deriv / np.linalg.norm(deriv)
    points = []
    points.append(point + norm)
    points.append(point - norm)
    points = np.asarray(points)
    return points

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
        plt.plot(T, BasisFirstDerivativeGraph(Knots, order, i, T))

    plt.show()

    for i in range(countPoints):
        plt.plot(T, BasisSecondDerivativeGraph(Knots, order, i, T))

    plt.show()

    BoxOrder = 3
    BoxPoints = np.array([[2, 0], [4, 0], [4, 2], [4, 4], [2, 4], [0, 4], [0, 2], [0, 0], [2, 0]])
    BoxKnots = OpenUniformKnotVector(BoxOrder, np.size(BoxPoints, 0), True)
    BoxT = np.arange(0., 1., 0.01)

    Box = BSpline(BoxPoints, BoxOrder, BoxKnots, BoxT)
    BoxFirstDerivative = BSplineFirstDerivative(BoxPoints, BoxOrder, BoxKnots, BoxT)
    BoxSecondDerivative = BSplineSecondDerivative(BoxPoints, BoxOrder, BoxKnots, BoxT)

    plt.plot(BoxPoints[:, 0], BoxPoints[:, 1])
    plt.plot(Box[:, 0], Box[:, 1])
    plt.plot(BoxFirstDerivative[:, 0], BoxFirstDerivative[:, 1])
    plt.plot(BoxSecondDerivative[:, 0], BoxSecondDerivative[:, 1])

    plt.show()

    BoxTangent = BSplineTangent(BoxPoints, BoxOrder, BoxKnots, 0.4)
    plt.plot(BoxPoints[:, 0], BoxPoints[:, 1])
    plt.plot(Box[:, 0], Box[:, 1])
    plt.plot(BoxTangent[:, 0], BoxTangent[:, 1])

    plt.show()