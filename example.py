import casadi as ca
import numpy as np
import time

# These parameters correspond to Table 1
T = 10
dt = 5 * 60
times = np.arange(dt, T * dt, dt)
H_b = -1.0
l = 1000.0
w = 5.0
C = 10.0
H_nominal = -0.25
Q_nominal = 0.5
Q0 = ca.DM([0, 0])
H0 = ca.DM([0, 0])

# Generic constants
g = 9.81
eps = 1e-12

# Derived quantities
dx = l / 2.0
A_nominal = w * (H_nominal - H_b)
P_nominal = w + 2 * (H_nominal - H_b)

# Smoothed absolute value function
sabs = lambda x : ca.sqrt(x**2 + eps)

# Symbols
Q = ca.MX.sym('Q', 2, T)
H = ca.MX.sym('H', 2, T)
theta = ca.MX.sym('theta')

# Left boundary condition
Q_left = ca.transpose(ca.DM(np.linspace(0.0, 1.0, T + 1)))

# Hydraulic constraints
Q_full = ca.horzcat(ca.vertcat(Q_left[0], Q0), ca.vertcat(Q_left[1:], Q))
H_full = ca.horzcat(H0, H)
A_full = w * 0.5 * (H_full[1:,:] + H_full[:-1,:] - 2 * H_b)
P_full = w + (H_full[1:,:] + H_full[:-1,:] - 2 * H_b)

c = w * (H_full[:,1:] - H_full[:,:-1]) / dt + (Q_full[1:,1:] - Q_full[:-1,1:]) / dx
d = (Q_full[1:-1,1:] - Q_full[1:-1,:-1]) / dt + g * (theta * A_full[:,1:] + (1 - theta) * A_nominal) * (H_full[1:,1:] - H_full[:-1,1:]) / dx + g * (theta * P_full[:,:-1] * sabs(Q_full[1:-1,1:]) / A_full[:,:-1]**2 + (1 - theta) * P_nominal * sabs(Q_nominal) / A_nominal**2) * Q_full[1:-1,1:] / C**2

# Objective function
f = ca.sum1(ca.vec(H[:, 1:]**2))

# Variable bounds
lbQ = ca.repmat(ca.DM([-np.inf, 0.0]), 1, T)
ubQ = ca.repmat(ca.DM([np.inf, 1.0]), 1, T)
lbH = ca.repmat(H_b, 2, T)
ubH = ca.repmat(np.inf, 2, T)

# Optimization problem
assert Q.size() == lbQ.size()
assert Q.size() == ubQ.size()
assert H.size() == lbH.size()
assert H.size() == ubH.size()

X = ca.veccat(Q, H)
lbX = ca.veccat(lbQ, lbH)
ubX = ca.veccat(ubQ, ubH)

g = ca.veccat(c, d)
lbg = ca.repmat(0, g.size1())
ubg = lbg

nlp = {'f': f, 'g': g, 'x': X, 'p': theta}
solver = ca.nlpsol('nlpsol', 'ipopt', nlp, {'ipopt': {'tol': 1e-10, 'constr_viol_tol': 1e-10, 'acceptable_tol': 1e-10, 'acceptable_constr_viol_tol': 1e-10, 'print_level': 0, 'print_timing_statistics': 'no', 'fixed_variable_treatment': 'make_constraint'}})

# Initial guess
x0 = ca.repmat(0, X.size1())

# Solve
t1 = time.time()

results = {}

theta_values = np.linspace(0.0, 1.0, 10)
for theta_value in theta_values:
    solution = solver(lbx = lbX, ubx = ubX, lbg = lbg, ubg = ubg, p = theta_value, x0 = x0)
    if solver.stats()['return_status'] != 'Solve_Succeeded':
        raise Exception('Solve failed with status {}'.format(solver.stats()['return_status']))
    x0 = solution['x']
    Q_res = ca.reshape(x0[:Q.size1() * Q.size2()], Q.size1(), Q.size2())
    H_res = ca.reshape(x0[Q.size1() * Q.size2():], H.size1(), H.size2())
    d = {}
    d['Q_1'] = Q_left
    d['Q_2'] = ca.horzcat(Q0[0], Q_res[0, :])
    d['Q_3'] = ca.horzcat(Q0[1], Q_res[1, :])
    d['H_1'] = ca.horzcat(H0[0], H_res[0, :])
    d['H_2'] = ca.horzcat(H0[1], H_res[1, :])
    results[theta_value] = d

t2 = time.time()

print("Time elapsed in solver: {}s".format(t2 - t1))

# Generate plots using matplotlib
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

X, Y = np.meshgrid(theta_values, times)
Z = np.zeros(X.shape)

for key in ['H_1', 'H_2', 'Q_1', 'Q_2', 'Q_3']:
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = results[X[i, j]][key][i]

    if key.startswith('Q'):
        unit = 'm$^3$/s'
    else:
        unit = 'm'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$\theta$ [-]')
    ax.set_ylabel(r'$t$ [$\cdot 10^3$ s]')
    ax.set_zlabel('${}$ []'.format(key, unit))
    ax.set_xlim(0, 1)
    ax.set_ylim(times[0], times[-1])

    if key.startswith('Q'):
        ax.set_zlim(0, 1)
    else:
        ax.set_zlim(-0.5, 0.5)

    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='black')

    ax.set_yticklabels(('{:.1f}'.format(ytick / 1.e3) for ytick in ax.get_yticks()))

    plt.savefig('{}.pdf'.format(key))
