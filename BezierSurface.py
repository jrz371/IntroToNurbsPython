import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from Bezier import BernsteinBasis

def BezierSurface(Points, UV):
    pointsU = np.size(Points, 0)
    pointsV = np.size(Points, 1)
    dimension = np.size(Points, 2)

    n = pointsU - 1
    m = pointsV - 1

    countU = np.size(UV, 0)
    countV = np.size(UV, 1)

    Out = np.zeros([countU, countV, dimension])

    for u in range(countU):
        for v in range(countV):
            value = np.zeros(dimension)
            uv = UV[u, v, :]
            for i in range(pointsU):
                for j in range(pointsV):
                    value += Points[i, j, :] * BernsteinBasis(n, i, uv[0]) * BernsteinBasis(m, j, uv[1])
            Out[u, v, :] = value
    return Out
    

if __name__ == "__main__":
    Points = np.asarray([
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [2, 1, 1], [3, 1, 0]],
        [[0, 2, 0], [1, 2, 1], [2, 2, 1], [3, 2, 0]],
        [[0, 3, 0], [1, 3, 0], [2, 3, 0], [3, 3, 0]],
    ])

    U = np.arange(0., 1., 0.05)
    V = np.arange(0., 1., 0.05)

    UV = np.transpose(np.asarray(np.meshgrid(U, V, sparse=False)))

    BezierSrf = BezierSurface(Points, UV)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_wireframe(Points[:, :, 0], Points[:, :, 1], Points[:, :, 2])
    ax.plot_surface(BezierSrf[:, :, 0], BezierSrf[:, :, 1], BezierSrf[:, :, 2])
    plt.show()