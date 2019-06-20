import time

from typing import Dict, Set

import casadi as ca
import numpy as np

# These parameters correspond to Table 1
T = 72
dt = 10 * 60
times = np.arange(0, (T + 1) * dt, dt)
n_level_nodes = 10
H_b = np.linspace(-4.9, -5.1, n_level_nodes)
l = 20000.0
w = 50.0
C = 40.0
H_nominal = 0.0
Q_nominal = 100
n_theta_steps = 10

# Generic constants
g = 9.81
eps = 1e-12
K = 10

# Derived quantities
dx = l / n_level_nodes
A_nominal = w * (H_nominal - np.mean(H_b))
P_nominal = w + 2 * (H_nominal - np.mean(H_b))

# Smoothed absolute value function
sabs = lambda x: ca.sqrt(x ** 2 + eps)

# Smoothed Heaviside function
sH = lambda x: 1 / (1 + ca.exp(-K * x))

# Compute steady state initial condition
Q0 = np.full(n_level_nodes, 100.0)
H0 = np.full(n_level_nodes, 0.0)
for i in range(1, n_level_nodes):
    A = w / 2.0 * (H0[i - 1] - H_b[i - 1] + H0[i] - H_b[i])
    P = w + H0[i - 1] - H_b[i - 1] + H0[i] - H_b[i]
    H0[i] = H0[i - 1] - dx * (P / A ** 3) * sabs(Q0[i]) * Q0[i] / C ** 2

# Symbols
Q = ca.MX.sym("Q", n_level_nodes, T)
H = ca.MX.sym("H", n_level_nodes, T)
delta = ca.MX.sym("delta", 1, T)
theta = ca.MX.sym("theta")

# Left boundary condition
Q_left = np.full(T + 1, 100)
Q_left[T // 3 : 2 * T // 3] = 300.0
Q_left = ca.DM(Q_left).T

# Hydraulic constraints
Q_full = ca.vertcat(Q_left, ca.horzcat(Q0, Q))
H_full = ca.horzcat(H0, H)
A_full = w * 0.5 * (H_full[1:, :] - H_b[1:] + H_full[:-1, :] - H_b[:-1])
A_full = ca.vertcat(w * (H_full[0, :] - H_b[0]), A_full, w * (H_full[-1, :] - H_b[-1]))
P_full = w + (H_full[1:, :] - H_b[1:] + H_full[:-1, :] - H_b[:-1])

c = w * (H_full[:, 1:] - H_full[:, :-1]) / dt + (Q_full[1:, 1:] - Q_full[:-1, 1:]) / dx
d = (
    (Q_full[1:-1, 1:] - Q_full[1:-1, :-1]) / dt
    + theta
    * (
        sH(Q_full[1:-1, :-1])
        * (
            Q_full[1:-1, :-1] ** 2 / A_full[1:-1, :-1]
            - Q_full[0:-2, :-1] ** 2 / A_full[0:-2, :-1]
        )
        / dx
        + (1 - sH(Q_full[1:-1, :-1]))
        * (
            Q_full[2:, :-1] ** 2 / A_full[2:, :-1]
            - Q_full[1:-1, :-1] ** 2 / A_full[1:-1, :-1]
        )
        / dx
    )
    + g
    * (theta * A_full[1:-1, 1:] + (1 - theta) * A_nominal)
    * (H_full[1:, 1:] - H_full[:-1, 1:])
    / dx
    + g
    * (
        theta * P_full[:, :-1] * sabs(Q_full[1:-1, 1:]) / A_full[1:-1, :-1] ** 2
        + (1 - theta) * P_nominal * sabs(Q_nominal) / A_nominal ** 2
    )
    * Q_full[1:-1, 1:]
    / C ** 2
)

# Integer constraint
control_indices = [i * n_level_nodes + n_level_nodes - 1 for i in range(T)]
e = Q[control_indices] - (100 + 100 * ca.transpose(delta))

# Objective function
f = ca.sum1(ca.vec(H ** 2))

# Variable bounds
lbQ = np.full(n_level_nodes, -np.inf)
# lbQ[-1] = 100.0
ubQ = np.full(n_level_nodes, np.inf)
# ubQ[-1] = 200.0

lbQ = ca.repmat(ca.DM(lbQ), 1, T)
ubQ = ca.repmat(ca.DM(ubQ), 1, T)
H_b_convolved_max = [
    max(H_b[np.array((max(0, i - 1), i, min(i + 1, H_b.size - 1)))])
    for i in range(len(H_b))
]
lbH = ca.repmat(H_b_convolved_max, 1, T)
ubH = ca.repmat(np.inf, n_level_nodes, T)

lbdelta = np.full(1, 0)
lbdelta = ca.repmat(ca.DM(lbdelta), 1, T)

ubdelta = np.full(1, 1)
ubdelta = ca.repmat(ca.DM(ubdelta), 1, T)

# Optimization problem
assert Q.size() == lbQ.size()
assert Q.size() == ubQ.size()
assert H.size() == lbH.size()
assert H.size() == ubH.size()

X = ca.veccat(Q, H, delta)
lbX = ca.veccat(lbQ, lbH, lbdelta)
ubX = ca.veccat(ubQ, ubH, ubdelta)

g = ca.veccat(c, d, e)
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

# Output problem structure
# TODO @Jakub
x0 = ca.repmat(0, X.size1())

print(lbX.toarray())
print(ubX.toarray())

f_g = ca.Function("g", [X, theta], [g])
g0 = ca.Function("g0", [X], [f_g(X, 0.0)])
Jg0 = g0.jacobian()
g0_x0 = g0(x0)
Jg0_x0 = Jg0(x0, g0_x0).sparse()

print(Jg0_x0)
print(lbg.toarray())
print(ubg.toarray())

f_f = ca.Function("f", [X, theta], [f])
f0 = ca.Function("f0", [X], [f_f(X, 0.0)])
Jf0 = f0.jacobian()
Hf0 = Jf0.jacobian()
f0_x0 = f0(x0)
Jf0_x0 = Jf0(x0, f0_x0).sparse()
Hf0_x0 = Hf0(x0, f0_x0, Jf0_x0).sparse()

print(Jf0_x0)
print(Hf0_x0)

binary_indices = range(Q.numel() + H.numel(), Q.numel() + H.numel() + delta.numel())

print(binary_indices)

# Solve
w_lbX = np.array(lbX, copy=True)
w_ubX = np.array(ubX, copy=True)


def homotopy_solve(delta_on: Set[int], delta_off: Set[int]) -> Dict[str, np.array]:
    # Fix integer variables
    offset = Q.numel() + H.numel()
    for i in range(delta.numel()):
        if i in delta_on:
            w_lbX[offset] = 1.0
            w_ubX[offset] = 1.0
        elif i in delta_off:
            w_lbX[offset] = 0.0
            w_ubX[offset] = 0.0
        else:
            w_lbX[offset] = 0.0
            w_ubX[offset] = 1.0
        offset += 1

    # Homotopy loop
    results = {}
    x0 = ca.repmat(0, X.size1())
    for theta_value in np.linspace(0.0, 1.0, n_theta_steps):
        solution = solver(lbx=w_lbX, ubx=w_ubX, lbg=lbg, ubg=ubg, p=theta_value, x0=x0)
        if solver.stats()["return_status"] != "Solve_Succeeded":
            raise Exception(
                "Solve failed with status {}".format(solver.stats()["return_status"])
            )
        x0 = solution["x"]
        Q_res = ca.reshape(x0[: Q.numel()], Q.size1(), Q.size2())
        H_res = ca.reshape(x0[Q.numel() : Q.numel() + H.numel()], H.size1(), H.size2())
        d = {}
        d["Q_0"] = np.array(Q_left).flatten()
        for i in range(n_level_nodes):
            d[f"Q_{i + 1}"] = np.array(ca.horzcat(Q0[i], Q_res[i, :])).flatten()
            d[f"H_{i + 1}"] = np.array(ca.horzcat(H0[i], H_res[i, :])).flatten()
        results[theta_value] = d
    return results


t0 = time.time()
results = homotopy_solve(set(), set())
print("Time elapsed in solver: {}s".format(time.time() - t0))
