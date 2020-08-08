import numpy as np
import matplotlib.pyplot as plt

def Combination(n, i):
    numerator = np.math.factorial(n)
    denomenator = np.math.factorial(i) * np.math.factorial(n - i)
    return numerator / denomenator

def BernsteinBasis(n, i, t):
    return Combination(n, i) * np.math.pow(t, i) * np.math.pow(1.0 - t, n - i)

def BezierCurve(T, Points):
    countPoints = np.size(Points, 0)
    n = countPoints - 1
    dim = np.size(Points, 1)
    rVal = []
    for t in T:
        point = np.zeros(dim)
        for i in range(0, countPoints):
            point += Points[i] * BernsteinBasis(n, i, t)
        rVal.append(point)
    return np.array(rVal)

if __name__ == "__main__":
    #Points = np.array([[0., -1.], [1., 1.], [2., -1.], [3., 1.]]) #Sawtooth Pattern
    #Points = np.array([[0., 0.], [2., 1.], [3., 1.], [5., 0.]]) #arc
    Points = np.array([[0., 0.], [3., 0.], [2., 2.], [1., 1.]]) #hook
    T = np.arange(0, 1, 0.01)

    Curve = BezierCurve(T, Points)

    plt.plot(Points[:, 0], Points[:, 1])
    plt.plot(Curve[:, 0], Curve[:, 1])
    plt.show()