import numpy as np
import matplotlib.pyplot as plt
from Bezier_1 import Combination

def CoefficentsMatrix(n):
    countPoints = n + 1
    m = np.zeros([countPoints, countPoints])
    nm = countPoints
    ni = 0
    for i in range(countPoints):
        for j in range(nm):
            m[i, j] = Combination(n, j) * Combination(n - j, n - i - j) * np.math.pow(-1, n - i - j)
        nm -= 1
        ni += 1
    return m

def MatrixBezier(Points, T):
    countPoints = np.size(Points, 0)
    n = countPoints - 1
    Coefficents = CoefficentsMatrix(n)
    power = np.flip(np.arange(0, countPoints, 1))
    tMat = np.tile(np.reshape(T, [len(T), 1]), [1, countPoints])
    tMat = np.power(tMat, power)
    return tMat.dot(Coefficents).dot(Points)

if __name__ == "__main__":
    Points = np.array([[0., 0.], [3., 0.], [2., 2.], [1., 1.]]) #hook
    T = np.arange(0, 1, 0.01)

    R = MatrixBezier(Points, T)

    plt.plot(Points[:, 0], Points[:, 1])
    plt.plot(R[:, 0], R[:, 1])
    plt.show()