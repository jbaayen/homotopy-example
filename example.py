import time

import casadi as ca
import numpy as np

# These parameters correspond to Table 1
T = 72
dt = 10 * 60
times = np.arange(0, (T + 1) * dt, dt)
n_level_nodes = 10
H_b = -5.0
l = 10000.0
w = 50.0
C = 40.0
H_nominal = 0.0
Q_nominal = 100
n_theta_steps = 10

# Generic constants
g = 9.81
eps = 1e-12

# Derived quantities
dx = l / n_level_nodes
A_nominal = w * (H_nominal - H_b)
P_nominal = w + 2 * (H_nominal - H_b)

# Smoothed absolute value function
sabs = lambda x: ca.sqrt(x ** 2 + eps)

# Compute steady state initial condition
Q0 = np.full(n_level_nodes, 100.0)
H0 = np.full(n_level_nodes, 0.0)
for i in range(1, n_level_nodes):
    A = w / 2.0 * (H0[i - 1] + H0[i] - 2 * H_b)
    P = w + H0[i - 1] + H0[i] - 2 * H_b
    H0[i] = H0[i - 1] - dx * (P / A ** 3) * sabs(Q0[i]) * Q0[i] / C ** 2

# Symbols
Q = ca.MX.sym("Q", n_level_nodes, T)
H = ca.MX.sym("H", n_level_nodes, T)
theta = ca.MX.sym("theta")

# Left boundary condition
Q_left = np.full(T + 1, 100)
Q_left[T // 4 : T // 2] = 300.0
Q_left = ca.DM(Q_left).T

# Hydraulic constraints
Q_full = ca.vertcat(Q_left, ca.horzcat(Q0, Q))
H_full = ca.horzcat(H0, H)
A_full = w * 0.5 * (H_full[1:, :] + H_full[:-1, :] - 2 * H_b)
P_full = w + (H_full[1:, :] + H_full[:-1, :] - 2 * H_b)

c = w * (H_full[:, 1:] - H_full[:, :-1]) / dt + (Q_full[1:, 1:] - Q_full[:-1, 1:]) / dx
d = (
    (Q_full[1:-1, 1:] - Q_full[1:-1, :-1]) / dt
    + g
    * (theta * A_full[:, 1:] + (1 - theta) * A_nominal)
    * (H_full[1:, 1:] - H_full[:-1, 1:])
    / dx
    + g
    * (
        theta * P_full[:, :-1] * sabs(Q_full[1:-1, 1:]) / A_full[:, :-1] ** 2
        + (1 - theta) * P_nominal * sabs(Q_nominal) / A_nominal ** 2
    )
    * Q_full[1:-1, 1:]
    / C ** 2
)

# Objective function
f = ca.sum1(ca.vec(H ** 2))

# Variable bounds
lbQ = np.full(n_level_nodes, -np.inf)
lbQ[-1] = 100.0
ubQ = np.full(n_level_nodes, np.inf)
ubQ[-1] = 200.0

lbQ = ca.repmat(ca.DM(lbQ), 1, T)
ubQ = ca.repmat(ca.DM(ubQ), 1, T)
lbH = ca.repmat(H_b, n_level_nodes, T)
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
            "tol": 1e-10,
            "constr_viol_tol": 1e-10,
            "acceptable_tol": 1e-10,
            "acceptable_constr_viol_tol": 1e-10,
            "print_level": 0,
            "print_timing_statistics": "no",
            "fixed_variable_treatment": "make_constraint",
        }
    },
)

# Initial guess
x0 = ca.repmat(0, X.size1())

# Solve
t0 = time.time()

results = {}

for theta_value in np.linspace(0.0, 1.0, n_theta_steps):
    solution = solver(lbx=lbX, ubx=ubX, lbg=lbg, ubg=ubg, p=theta_value, x0=x0)
    if solver.stats()["return_status"] != "Solve_Succeeded":
        raise Exception(
            "Solve failed with status {}".format(solver.stats()["return_status"])
        )
    x0 = solution["x"]
    Q_res = ca.reshape(x0[: Q.size1() * Q.size2()], Q.size1(), Q.size2())
    H_res = ca.reshape(x0[Q.size1() * Q.size2() :], H.size1(), H.size2())
    d = {}
    d["Q_1"] = np.array(Q_left).flatten()
    for i in range(n_level_nodes):
        d[f"Q_{i + 2}"] = np.array(ca.horzcat(Q0[i], Q_res[i, :])).flatten()
        d[f"H_{i + 1}"] = np.array(ca.horzcat(H0[i], H_res[i, :])).flatten()
    results[theta_value] = d

print("Time elapsed in solver: {}s".format(time.time() - t0))
