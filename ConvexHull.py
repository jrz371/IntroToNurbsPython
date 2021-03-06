import numpy as np
import matplotlib.pyplot as plt
from BSpline import BSpline, OpenUniformKnotVector
import math

def FindSide(P1, P2, P):
    val = (P[1] - P1[1]) * (P2[0] - P1[0]) - (P2[1] - P1[1]) * (P[0] - P1[0])

    return np.sign(val)

def LineDist(P1, P2, P):
    return abs((P[1] - P1[1]) * (P2[0] - P1[0]) - (P2[1] - P1[1]) * (P[0] - P1[0]))

def QuickHull(Points, n, P1, P2, side, Hull):
    index = -1
    max_dist = 0

    for i in range(0, n):
        temp = LineDist(P1, P2, Points[i, :])
        if(FindSide(P1, P2, Points[i, :]) == side and temp > max_dist):
            index = i
            max_dist = temp

    if(index == -1):
        Hull.append(P1)
        Hull.append(P2)
        return
    
    QuickHull(Points, n, Points[index, :], P1, -FindSide(Points[index, :], P1, P2), Hull)
    QuickHull(Points, n, Points[index, :], P2, -FindSide(Points[index, :], P2, P1), Hull)

def ConvexHull2D(Points):
    maxX = np.argmax(Points[:, 0])
    minX = np.argmin(Points[:, 0])
    countPoints = np.size(Points, 0)

    Hull = []

    QuickHull(Points, countPoints, Points[minX, :], Points[maxX, :], 1, Hull)
    QuickHull(Points, countPoints, Points[minX, :], Points[maxX, :], -1, Hull)

    Hull = np.asarray(Hull)

    Hull = np.unique(Hull, axis=0)

    #sort by angle so it plots nicely
    Center = np.average(Hull)

    ToCenter = Hull - Center
    Angles = np.arctan2(ToCenter[:, 0], ToCenter[:, 1])

    SortedAngles = np.argsort(Angles)

    Hull = np.array(Hull)[SortedAngles]

    #copy first element so it loops on itself in plot
    Hull = np.append(Hull, [Hull[0, :]], axis=0)

    return Hull

if __name__ == "__main__":
    #shows that the b spline curve fits in the convex hull of the control points
    Points = np.array([[1., 1.], [2., 4.], [2., 6.], [4., 3.], [6., 6.], [8., 6.]])
    countPoints = np.size(Points, 0)
    order = 3

    Knots = OpenUniformKnotVector(order, countPoints, True)
    T = np.arange(0, 1.0, 0.01)

    Spline = BSpline(Points, order, Knots, T)

    Hull = ConvexHull2D(Points)

    plt.plot(Points[:, 0], Points[:, 1])
    plt.plot(Hull[:, 0], Hull[:, 1], 'r--')
    plt.plot(Spline[:, 0], Spline[:, 1])
    plt.show()

    