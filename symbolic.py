import math
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

dims = ['x', 'y', 'z']

def binomial(n, i):
    return math.factorial(n) / (math.factorial(i) * math.factorial(n - i))

def symbolic_bernstein_basis(t, n, i):
    b = binomial(n, i)
    return b * sp.Pow(t, i) * sp.Pow(1.0 - t, n - i)

#non-rational bezier curve
#returns pts, axis equations
#pts[order][dim]
#axis_eq[dim]
def symbolic_bez_crv(t, dim, order):
    pts = []
    terms = []

    for i in range(order):
        pt = []
        terms.append([])

        for j in range(dim):
            name = 'p' + str(i) + dims[j]
            p = sp.Symbol(name)
            term = symbolic_bernstein_basis(t, order - 1, i) * p
            pt.append(p)

            terms[i].append(term)

        pts.append(pt)
    
    axis_eqs = []
    for i in range(dim):
        r = []
        for j in range(order):
            r.append(terms[j][i])
        axis_eqs.append(sum(r))

    return pts, axis_eqs

def eval_symbolic_bez_crv(order, symbolic_t, symbolic_control_pts, axis_eqs, t_vals, control_pts):
    dim = len(axis_eqs)
    
    sub_dict = {}

    for i in range(order):
        for j in range(dim):
            sub_dict[symbolic_control_pts[i][j]] = control_pts[i][j]
    
    evaluations = len(t_vals)
    eval_pts = np.zeros([evaluations, dim])

    for i in range(evaluations):
        sub_dict[symbolic_t] = t_vals[i]

        for j in range(dim):
            eval_pts[i][j] = axis_eqs[j].evalf(subs=sub_dict)

    return eval_pts


#non-rational bezier surface
#returns pts, axis equations
#pts[u][v][dim]
#axis_eq[dim]
def symbolic_bez_srf(u, v, order_u, order_v):
    dim = 3 #constant dimension

    pts = []
    terms = []

    for i in range(order_u):
        pts.append([])
        terms.append([])

        for j in range(order_v):
            
            pts[i].append([])
            terms[i].append([])
            
            for k in range(dim):
                name = 'p-' + str(i) + '-' + str(j) + dims[k]
                p = sp.Symbol(name)
                term = symbolic_bernstein_basis(u, order_u - 1, i) * symbolic_bernstein_basis(v, order_v - 1, j) * p

                pts[i][j].append(p)
                terms[i][j].append(term)
    
    axis_eqs = []
    for k in range(dim):
        r = []

        for i in range(order_u):
            for j in range(order_v):
                r.append(terms[i][j][k])
        
        axis_eqs.append(sum(r))

    return pts, axis_eqs

def plot_symbolic_bez_srf(order_u, order_v, symbol_u, symbol_v, control_pt_symbols, axis_eqs, control_pts):
    dim = 3
    sub_dict = {}

    for i in range(order_u):
        for j in range(order_v):
            for k in range(dim):
                sub_dict[control_pt_symbols[i][j][k]] = control_pts[i][j][k]

    u_params = np.arange(0., 1., 0.05)
    v_params = np.arange(0., 1., 0.05)

    count_u = len(u_params)
    count_v = len(v_params)

    evaluated_pts = np.zeros([count_u, count_v, 3])

    for i in range(count_u):
        for j in range(count_v):
            sub_dict[symbol_u] = u_params[i]
            sub_dict[symbol_v] = v_params[j]
            
            evaluated_pts[i, j, 0] = axis_eqs[0].evalf(subs=sub_dict)
            evaluated_pts[i, j, 1] = axis_eqs[1].evalf(subs=sub_dict)
            evaluated_pts[i, j, 2] = axis_eqs[2].evalf(subs=sub_dict)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_wireframe(control_pts[:, :, 0], control_pts[:, :, 1], control_pts[:, :, 2])
    ax.plot_surface(evaluated_pts[:, :, 0], evaluated_pts[:, :, 1], evaluated_pts[:, :, 2])
    plt.show()

if __name__ == "__main__":
    u = sp.symbols('u')
    v = sp.symbols('v')
    pts, axis_eqs = symbolic_bez_srf(u, v, 4, 4)
    
    control_points = np.asarray([
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [2, 1, 1], [3, 1, 0]],
        [[0, 2, 0], [1, 2, 1], [2, 2, 1], [3, 2, 0]],
        [[0, 3, 0], [1, 3, 0], [2, 3, 0], [3, 3, 0]],
    ])

    plot_symbolic_bez_srf(4, 4, u, v, pts, axis_eqs, control_points)
