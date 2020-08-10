import numpy as np
import matplotlib.pyplot as plt
from KnotVectors import OpenUniformKnotVector, PeriodicKnotVector
from BSpline import BSpline

#returns P which corresponds to multiplicity of each knot vector value
#and s which is the number of knot vector values that have multiple occurences
def KnotVectorMultiplicity(Knots):
    V, P = np.unique(Knots, False, False, True)
    s = np.sum(P[1:-1] > 1) #discard first/last since s only cares about internal values
    return V, P, s

def RaiseOrderTwoToThree(OriginalPoints, OriginalKnots):
    Values, P, s = KnotVectorMultiplicity(OriginalKnots)
    countPoints = np.size(OriginalPoints, axis=0)
    dimension = np.size(OriginalPoints, 1)
    countNewPoints = countPoints + 3 + s #3 is the new order
    NewPoints = np.zeros([countNewPoints, dimension])

    mpts = 0

    if OriginalKnots[0] < OriginalKnots[1]:
        NewPoints[0, :] = OriginalPoints[0, :] * 0.5
        mpts += 1
    
    NewPoints[mpts, :] = OriginalPoints[0, :]

    for i in range(1, countPoints):
        if OriginalKnots[i] < OriginalKnots[i + 1]:
            mpts += 1
            NewPoints[mpts, :] = (OriginalPoints[i, :] + OriginalPoints[i - 1, :]) * 0.5
        mpts += 1
        NewPoints[mpts, :] = OriginalPoints[i, :]
    
    if OriginalKnots[countPoints] < OriginalKnots[countPoints + 1]:
        mpts += 1
        NewPoints[mpts, :] = OriginalPoints[countPoints - 1, :] * 0.5

    NewKnots = np.asarray([])
    countP = len(P)
    for i in range(countP):
        NewKnots = np.append(NewKnots, np.ones(P[i] + 1) * Values[i], axis=0) #Duplicate knot values

    return NewPoints, NewKnots

if __name__ == "__main__":
    Points = np.array([[1., 1.], [2., 2.], [3., 2.], [4., 1.]])
    countPoints = np.size(Points, 0)
    oldOrder = 2
    Knots = OpenUniformKnotVector(oldOrder, countPoints, True)

    T = np.arange(0., 1., 0.01)

    SplineOld = BSpline(Points, oldOrder, Knots, T)
    
    plt.plot(Points[:, 0], Points[:, 1], "x")
    plt.plot(SplineOld[:, 0], SplineOld[:, 1])

    plt.show()

    newOrder = 3

    NewPoints, NewKnots = RaiseOrderTwoToThree(Points, Knots)

    SplineNew = BSpline(NewPoints, newOrder, NewKnots, T)

    plt.plot(NewPoints[:, 0], NewPoints[:, 1], "x")
    plt.plot(SplineNew[:, 0], SplineNew[:, 1])

    plt.show()