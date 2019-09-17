import time
import sys

import casadi as ca
import numpy as np


# These parameters correspond to Table 1
dt = 5 * 60
steps_per_hour = round(3600 / dt)
T = 48 * steps_per_hour
times = np.arange(0, (T + 1) * dt, dt)
n_level_nodes = 16
H_b = np.linspace(-5.0, -5.1875, n_level_nodes)
l = 10000.0
w = 50.0
C = 40.0
H_nominal = 0.0
Q_nominal = 100
n_theta_steps = 10
trace_path = True

# Generic constants
g = 9.81
eps = 1e-12
K = 10
alpha = 2
min_depth = 1e-2

# Derived quantities
dxH = ca.MX(ca.repmat(l / (n_level_nodes - 1), n_level_nodes - 1))
dxQ = ca.MX(ca.vertcat(0.5 * dxH[0], 0.5 * (dxH[1:] + dxH[:-1]), 0.5 * dxH[-1]))
A_nominal = w * (H_nominal - np.mean(H_b))
P_nominal = w + 2 * (H_nominal - np.mean(H_b))

# Smoothed absolute value function
sabs = lambda x: ca.sqrt(x ** 2 + eps)

# Smoothed Heaviside function
sH = lambda x: 1 / (1 + ca.exp(-K * x))

# Smoothed max function
smax = lambda x, y: (x * ca.exp(alpha * x) + y * ca.exp(alpha * y)) / (
    ca.exp(alpha * x) + ca.exp(alpha * y)
)

# Initial condition
Q0 = np.full(n_level_nodes, 100.0)
H0 = np.linspace(0.0, -0.26, n_level_nodes)

# Symbols
Q = ca.MX.sym("Q", n_level_nodes, T)
H = ca.MX.sym("H", n_level_nodes, T)
theta = ca.MX.sym("theta")

# Left boundary condition
Q_left = np.full(T + 1, 100)
Q_left[20 * steps_per_hour : (20 + 8) * steps_per_hour + 1] = 300.0
Q_left = ca.DM(Q_left).T

# Hydraulic constraints
Q_full = ca.vertcat(Q_left, ca.horzcat(Q0, Q))
H_full = ca.horzcat(H0, H)
depth_full = H_full - H_b
depth_full_c = smax(min_depth, depth_full)
A_full_H = w * depth_full_c
A_full_Q = ca.vertcat(
    A_full_H[0, :], 0.5 * (A_full_H[1:, :] + A_full_H[:-1, :]), A_full_H[-1, :]
)
P_full_Q = w + (depth_full_c[1:, :] + depth_full_c[:-1, :])

c = (
    theta * (A_full_H[:, 1:] - A_full_H[:, :-1])
    + (1 - theta) * w * (depth_full[:, 1:] - depth_full[:, :-1])
) / dt + (Q_full[1:, 1:] - Q_full[:-1, 1:]) / dxQ
d = (
    (Q_full[1:-1, 1:] - Q_full[1:-1, :-1]) / dt
    + theta
    * (
        2
        * Q_full[1:-1, :-1]
        / A_full_Q[1:-1, :-1]
        * (
            sH(Q_full[1:-1, :-1]) * (Q_full[1:-1, :-1] - Q_full[0:-2, :-1]) / dxQ[:-1]
            + (1 - sH(Q_full[1:-1, :-1]))
            * (Q_full[2:, :-1] - Q_full[1:-1, :-1])
            / dxQ[1:]
        )
        - (Q_full[1:-1, :-1] / A_full_Q[1:-1, :-1]) ** 2
        * (A_full_H[1:, :-1] - A_full_H[:-1, :-1])
        / dxH
    )
    + g
    * (theta * A_full_Q[1:-1, :-1] + (1 - theta) * A_nominal)
    * (H_full[1:, 1:] - H_full[:-1, 1:])
    / dxH
    + g
    * (
        theta * P_full_Q[:, :-1] * sabs(Q_full[1:-1, :-1]) / A_full_Q[1:-1, :-1] ** 2
        + (1 - theta) * P_nominal * sabs(Q_nominal) / A_nominal ** 2
    )
    * Q_full[1:-1, 1:]
    / C ** 2
)

# Objective function
f = ca.sum1(ca.vec(H[0, :] ** 2 + H[-1, :] ** 2))

# Variable bounds
lbQ = np.full(n_level_nodes, -np.inf)
lbQ[-1] = 0.0
ubQ = np.full(n_level_nodes, np.inf)
ubQ[-1] = 200.0

lbQ = ca.repmat(ca.DM(lbQ), 1, T)
ubQ = ca.repmat(ca.DM(ubQ), 1, T)
lbH = ca.repmat(-np.inf, n_level_nodes, T)
ubH = ca.repmat(np.inf, n_level_nodes, T)

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

nlp = {"f": f, "g": g, "x": X, "p": theta}
solver = ca.nlpsol(
    "nlpsol",
    "ipopt",
    nlp,
    {
        "ipopt": {
            "tol": 1e-6,
            "constr_viol_tol": 1e-6,
            "acceptable_tol": 1e-6,
            "acceptable_constr_viol_tol": 1e-6,
            "fixed_variable_treatment": "make_constraint",
        }
    },
)

# Initial guess
x0 = ca.repmat(0, X.size1())

# Solve
t0 = time.time()

results = {}


def solve(theta_value):
    global x0
    solution = solver(lbx=lbX, ubx=ubX, lbg=lbg, ubg=ubg, p=theta_value, x0=x0)
    if solver.stats()["return_status"] != "Solve_Succeeded":
        raise Exception(
            "Solve failed with status {}".format(solver.stats()["return_status"])
        )
    x0 = solution["x"]
    Q_res = ca.reshape(x0[: Q.size1() * Q.size2()], Q.size1(), Q.size2())
    H_res = ca.reshape(x0[Q.size1() * Q.size2() :], H.size1(), H.size2())
    d = {}
    d["Q_0"] = np.array(Q_left).flatten()
    for i in range(n_level_nodes):
        d[f"Q_{i + 1}"] = np.array(ca.horzcat(Q0[i], Q_res[i, :])).flatten()
        d[f"H_{i + 1}"] = np.array(ca.horzcat(H0[i], H_res[i, :])).flatten()
    results[theta_value] = d


if trace_path:
    for theta_value in np.linspace(0.0, 1.0, n_theta_steps):
        solve(theta_value)
else:
    solve(1.0)

print("Time elapsed in solver: {}s".format(time.time() - t0))

# Output to CSV
import pandas as pd

df = pd.DataFrame(data=results[1.0])
df.to_csv("results.csv")
