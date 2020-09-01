import time

import casadi as ca
import numpy as np

# These parameters correspond to Table 1
T = 72
dt = 10 * 60
times = np.arange(0, (T + 1) * dt, dt)
n_level_nodes = 10
H_b = np.linspace(-4.9, -5.1, n_level_nodes)
l = 10000.0
w = 50.0
C = 40.0
H_nominal = 0.0
Q_nominal = 100
Q_min = 100.0
Q_max = 200.0
n_theta_steps = 10
trace_path = False
multi_start = True
n_starting_points = 1000

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
        2
        * Q_full[1:-1, :-1]
        / A_full[1:-1, :-1]
        * (
            sH(Q_full[1:-1, :-1]) * (Q_full[1:-1, :-1] - Q_full[0:-2, :-1]) / dx
            + (1 - sH(Q_full[1:-1, :-1])) * (Q_full[2:, :-1] - Q_full[1:-1, :-1]) / dx
        )
        - Q_full[1:-1, :-1] ** 2
        / A_full[1:-1, :-1] ** 2
        * w
        * (H_full[1:, :-1] - H_full[:-1, :-1])
        / dx
    )
    + g
    * (theta * A_full[1:-1, :-1] + (1 - theta) * A_nominal)
    * (H_full[1:, 1:] - H_full[:-1, 1:])
    / dx
    + g
    * (
        theta * P_full[:, :-1] * sabs(Q_full[1:-1, :-1]) / A_full[1:-1, :-1] ** 2
        + (1 - theta) * P_nominal * sabs(Q_nominal) / A_nominal ** 2
    )
    * Q_full[1:-1, 1:]
    / C ** 2
)

# Objective function
f = ca.sum1(ca.vec(H ** 2))

# Variable bounds
lbQ = np.full(n_level_nodes, -np.inf)
lbQ[-1] = Q_min
ubQ = np.full(n_level_nodes, np.inf)
ubQ[-1] = Q_max

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

# Solve
t0 = time.time()

results = {}


def solve(theta_value, x0):
    solution = solver(lbx=lbX, ubx=ubX, lbg=lbg, ubg=ubg, p=theta_value, x0=x0)
    if solver.stats()["return_status"] != "Solve_Succeeded":
        raise Exception(
            "Solve failed with status {}".format(solver.stats()["return_status"])
        )
    x = solution["x"]
    Q_res = ca.reshape(x[: Q.size1() * Q.size2()], Q.size1(), Q.size2())
    H_res = ca.reshape(x[Q.size1() * Q.size2() :], H.size1(), H.size2())
    d = {}
    d["Q_0"] = np.array(Q_left).flatten()
    for i in range(n_level_nodes):
        d[f"Q_{i + 1}"] = np.array(ca.horzcat(Q0[i], Q_res[i, :])).flatten()
        d[f"H_{i + 1}"] = np.array(ca.horzcat(H0[i], H_res[i, :])).flatten()
    results[theta_value] = d
    return x


if multi_start:
    import pyDOE

    starting_points = pyDOE.lhs(X.size1(), n_starting_points)
    starting_points[:, : Q.size1() * Q.size2()] *= Q_max - Q_min
    starting_points[:, : Q.size1() * Q.size2()] += Q_min
else:
    starting_points = ca.repmat(0, X.size1())

solutions = []
for i, x0 in enumerate(starting_points):
    print(f"{i}/{len(starting_points)}")
    if trace_path:
        for theta_value in np.linspace(0.0, 1.0, n_theta_steps):
            x0 = solve(theta_value, x0)
    else:
        x0 = solve(1.0, x0)
    solutions.append(x0.toarray().flatten())

print("Time elapsed in solver: {}s".format(time.time() - t0))

solutions = np.array(solutions)
print(list(np.std(solutions, axis=0)))
