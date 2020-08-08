import numpy as np
import matplotlib.pyplot as plt
from KnotVectors import OpenUniformKnotVector
from BSplineBasis import CoxDeBoorRecursion
from BSpline import BSpline

def PointsToCurve(PointsToFit, countControlPoints, order):
    Knots = OpenUniformKnotVector(order, countControlPoints, True)

    Distances = [0.,]
    countPoints = np.size(PointsToFit, 0)
    distance = 0.

    for i in range(countPoints - 1):
        distance += np.sqrt(np.sum(np.power(PointsToFit[i + 1, :] - PointsToFit[i, :], 2.)))
        Distances.append(distance)

    Distances = np.asarray(Distances)
    Distances /= np.max(Distances)
    Distances *= 1.0 - 1e-5 # 0 <= T < 1, subtract small epsilon so point exists on the curve

    NMatrix = np.zeros([countPoints, countControlPoints])

    for i in range(countPoints):
        for j in range(countControlPoints):
            NMatrix[i, j] = CoxDeBoorRecursion(j, order, Knots, Distances[i])

    NInverse = None

    if countPoints == countControlPoints:
        NInverse = np.linalg.inv(NMatrix)
    else:
        NInverse = np.linalg.pinv(NMatrix)

    Points = np.dot(NInverse, PointsToFit)

    return Knots, Points

if __name__ == "__main__":
    PointsFit = np.asarray([[0, 0], [1.5, 2], [3, 2.5], [4.5, 2], [6, 0]])

    order = 3
    countPoints = 5

    Knots, Points = PointsToCurve(PointsFit, countPoints, order)
    
    T = np.arange(0., 1., 0.01)

    Spline = BSpline(Points, order, Knots, T)

    plt.plot(Points[:, 0], Points[:, 1])
    plt.plot(Spline[:, 0], Spline[:, 1])
    plt.plot(PointsFit[:, 0], PointsFit[:, 1], "x")
    plt.show()
