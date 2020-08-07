import numpy as np
import matplotlib.pyplot as plt
from BezierMatrix import MatrixBezier
from Bezier import BernsteinBasis

def BernsteinFirstDerivative(n, i ,t):
    return (i - n * t) / (t * (1. - t)) * BernsteinBasis(n, i, t)

def BernsteinSecondDerivative(n, i, t):
    numerator = pow(i - n * t, 2) - n * pow(t, 2) - i * (1.0 - 2 * t)
    denominator = pow(t, 2) * pow(1 - t, 2)
    return numerator / denominator * BernsteinBasis(n, i, t)

def BezierFirstDerivative(Points, T):
    countPoints = np.size(Points, 0)
    n = countPoints - 1
    dimension = np.size(Points, 1)
    rVal = []
    for t in T:
        evaluatedPoint = np.zeros(dimension)
        for i in range(countPoints):
            evaluatedPoint += Points[i] * BernsteinFirstDerivative(n, i, t)
        rVal.append(evaluatedPoint)
    return np.array(rVal)

def BezierSecondDerivative(Points, T):
    countPoints = np.size(Points, 0)
    n = countPoints - 1
    dimension = np.size(Points, 1)
    rVal = []
    for t in T:
        evaluatedPoint = np.zeros(dimension)
        for i in range(countPoints):
            evaluatedPoint += Points[i] * BernsteinSecondDerivative(n, i, t)
        rVal.append(evaluatedPoint)
    return np.array(rVal)

def BezierTangent(Points, t):
    deriv = BezierFirstDerivative(Points, [t])[0]
    point = MatrixBezier(Points, [t])[0]
    norm = deriv / np.linalg.norm(deriv)
    points = []
    points.append(point + norm)
    points.append(point - norm)
    points = np.asarray(points)
    return points

if __name__ == "__main__":
    Points = np.array([[1., 1.], [2., 3.], [4., 3.], [3., 1.]])
    T = np.arange(0, 1, 0.01)

    R = MatrixBezier(Points, T)

    R_deriv = BezierFirstDerivative(Points, T)
    R_deriv2 = BezierSecondDerivative(Points, T)

    Tangent = BezierTangent(Points, 0.35)

    plt.plot(R_deriv[:, 0], R_deriv[:, 1])
    plt.plot(R_deriv2[:, 0], R_deriv2[:, 1])
    plt.plot(Points[:, 0], Points[:, 1])
    plt.plot(Tangent[:, 0], Tangent[:, 1])
    plt.plot(R[:, 0], R[:, 1])
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.show()