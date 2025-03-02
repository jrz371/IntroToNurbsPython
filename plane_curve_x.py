import math
import sympy as sp
from symbolic import *

if __name__ == "__main__":
    #plane equation
    plane_eqn = [0,0,1,0]
    a, b, c, d = sp.symbols('a b c d')

    t = sp.symbols('t')

    pts, axis_eqs = symbolic_bez_crv(t, 3, 3)

    control_pts = np.array([
        [-1, 0, -1],
        [0, 0, 2],
        [1, 0, -1]
    ])

    vals = {
        a: plane_eqn[0],
        b: plane_eqn[1],
        c: plane_eqn[2],
        d: plane_eqn[3],
        pts[0][0]: control_pts[0][0],
        pts[0][1]: control_pts[0][1],
        pts[0][2]: control_pts[0][2],
        pts[1][0]: control_pts[1][0],
        pts[1][1]: control_pts[1][1],
        pts[1][2]: control_pts[1][2],
        pts[2][0]: control_pts[2][0],
        pts[2][1]: control_pts[2][1],
        pts[2][2]: control_pts[2][2],
    }

    plane_crv_csx = a * axis_eqs[0] + b * axis_eqs[1] + c * axis_eqs[2] + d

    solutions = sp.solve(plane_crv_csx, t)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    #plane
    x, y = np.meshgrid(range(-3, 3), range(-3, 3))
    z = (-plane_eqn[0] * x - plane_eqn[1] * y - plane_eqn[3]) * 1.0 /plane_eqn[2]
    ax.plot_surface(x, y, z, alpha=0.2)

    #crv
    t_vals = np.arange(0, 1, 0.01)
    eval_pts = eval_symbolic_bez_crv(3, t, pts, axis_eqs, t_vals, control_pts)
    ax.plot(eval_pts[:, 0], eval_pts[:, 1], eval_pts[:, 2])

    #intersection points
    solution_params = [sol.evalf(subs=vals) for sol in solutions]
    solution_pts = eval_symbolic_bez_crv(3, t, pts, axis_eqs, solution_params, control_pts)

    ax.scatter(solution_pts[:, 0], solution_pts[:, 1], solution_pts[:, 2])

    plt.show()