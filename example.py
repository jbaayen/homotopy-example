# Example code illustrating homotopy - CPLEX coupling for global mixed-integer non-linear optimization
#
# Copyright (C) 2017 - 2019 Jorn Baayen
# Copyright (C) 2019 KISTERS AG
# Copyright (C) 2019 IBM
#
# Authors:  Jorn Baayen, Jakub Marecek

import time, sys
import casadi as ca
import numpy as np

import cplex
from cplex.exceptions import CplexError
from cplex.callbacks import IncumbentCallback, BranchCallback


class OptimizationError(Exception):
    pass


class ChannelModel:
    def __init__(self, n_weirs, n_level_nodes, n_time_steps, dt):
        assert n_level_nodes % n_weirs == 0

        # These parameters correspond to Table 1
        dt = 10 * 60
        self.times = np.arange(0, (n_time_steps + 1) * dt, dt)
        self.n_level_nodes = n_level_nodes
        n_level_nodes_per_reach = n_level_nodes // n_weirs
        reach_length = 10000.0
        w = 50.0
        C = 40.0
        H_nominal = 0.0
        Q_nominal = 100
        self.n_theta_steps = 2

        # Bottom level
        dx = reach_length / n_level_nodes_per_reach
        dydx = -0.2 / reach_length
        H_b = np.tile(
            np.linspace(
                -5.0 + dydx * 0.5 * dx,
                -5.0 + dydx * reach_length - dydx * 0.5 * dx,
                n_level_nodes_per_reach,
            ),
            n_weirs,
        )

        # Generic constants
        g = 9.81
        eps = 1e-12
        K = 10
        alpha = 20
        min_depth = 1e-3

        # Derived quantities
        A_nominal = w * (H_nominal - np.mean(H_b))
        P_nominal = w + 2 * (H_nominal - np.mean(H_b))

        dxH = ca.MX(
            ca.repmat(reach_length / n_level_nodes_per_reach, self.n_level_nodes - 1)
        )
        dxQ = ca.MX(
            ca.repmat(reach_length / n_level_nodes_per_reach, self.n_level_nodes)
        )

        # Smoothed absolute value function
        sabs = lambda x: ca.sqrt(x ** 2 + eps)

        # Smoothed max
        smax = lambda x, y: (x * ca.exp(alpha * x) + y * ca.exp(alpha * y)) / (
            ca.exp(alpha * x) + ca.exp(alpha * y)
        )

        # Smoothed Heaviside function
        sH = lambda x: 1 / (1 + ca.exp(-K * x))

        # Compute steady state initial condition
        Q0 = np.full(n_level_nodes_per_reach, 100.0)
        H0 = np.full(n_level_nodes_per_reach, 0.0)
        for i in range(1, n_level_nodes_per_reach):
            A = w / 2.0 * (H0[i - 1] - H_b[i - 1] + H0[i] - H_b[i])
            P = w + H0[i - 1] - H_b[i - 1] + H0[i] - H_b[i]
            H0[i] = H0[i - 1] - dx * (P / A ** 3) * sabs(Q0[i]) * Q0[i] / C ** 2
        self.Q0 = np.tile(Q0, n_weirs)
        self.H0 = np.tile(H0, n_weirs)

        # Symbols
        self.Q = ca.MX.sym("Q", self.n_level_nodes, n_time_steps)
        self.H = ca.MX.sym("H", self.n_level_nodes, n_time_steps)
        self.delta = ca.MX.sym("delta", n_weirs, n_time_steps)
        self.theta = ca.MX.sym("theta")

        # Left boundary condition
        self.Q_left = np.full(n_time_steps + 1, 100)
        self.Q_left[n_time_steps // 3 : 2 * n_time_steps // 3] = 300.0
        self.Q_left = ca.DM(self.Q_left).T

        # Hydraulic constraints
        Q_full = ca.vertcat(self.Q_left, ca.horzcat(self.Q0, self.Q))
        H_full = ca.horzcat(self.H0, self.H)
        depth_full = H_full - H_b
        depth_full_c = smax(min_depth, depth_full)
        A_full_H = w * depth_full_c
        A_full_Q = ca.vertcat(
            A_full_H[0, :], 0.5 * (A_full_H[1:, :] + A_full_H[:-1, :]), A_full_H[-1, :]
        )
        P_full_H = w + 2 * depth_full_c
        P_full_Q = ca.vertcat(
            P_full_H[0, :], 0.5 * (P_full_H[1:, :] + P_full_H[:-1, :]), P_full_H[-1, :]
        )

        c = (
            self.theta * (A_full_H[:, 1:] - A_full_H[:, :-1])
            + (1 - self.theta) * w * (depth_full[:, 1:] - depth_full[:, :-1])
        ) / dt + (Q_full[1:, 1:] - Q_full[:-1, 1:]) / dxQ
        d = ca.vertcat(
            *(
                self.theta
                * (
                    2
                    * Q_full[
                        weir * n_level_nodes_per_reach
                        + 1 : (weir + 1) * n_level_nodes_per_reach,
                        :-1,
                    ]
                    / A_full_Q[
                        weir * n_level_nodes_per_reach
                        + 1 : (weir + 1) * n_level_nodes_per_reach,
                        :-1,
                    ]
                    * (
                        sH(
                            Q_full[
                                weir * n_level_nodes_per_reach
                                + 1 : (weir + 1) * n_level_nodes_per_reach,
                                :-1,
                            ]
                        )
                        * (
                            Q_full[
                                weir * n_level_nodes_per_reach
                                + 1 : (weir + 1) * n_level_nodes_per_reach,
                                :-1,
                            ]
                            - Q_full[
                                weir
                                * n_level_nodes_per_reach : (weir + 1)
                                * n_level_nodes_per_reach
                                - 1,
                                :-1,
                            ]
                        )
                        / dxQ[
                            weir
                            * n_level_nodes_per_reach : (weir + 1)
                            * n_level_nodes_per_reach
                            - 1
                        ]
                        + (
                            1
                            - sH(
                                Q_full[
                                    weir * n_level_nodes_per_reach
                                    + 1 : (weir + 1) * n_level_nodes_per_reach,
                                    :-1,
                                ]
                            )
                        )
                        * (
                            Q_full[
                                weir * n_level_nodes_per_reach
                                + 2 : (weir + 1) * n_level_nodes_per_reach
                                + 1,
                                :-1,
                            ]
                            - Q_full[
                                weir * n_level_nodes_per_reach
                                + 1 : (weir + 1) * n_level_nodes_per_reach,
                                :-1,
                            ]
                        )
                        / dxQ[
                            weir * n_level_nodes_per_reach
                            + 1 : (weir + 1) * n_level_nodes_per_reach
                        ]
                    )
                    - (
                        Q_full[
                            weir * n_level_nodes_per_reach
                            + 1 : (weir + 1) * n_level_nodes_per_reach,
                            :-1,
                        ]
                        / A_full_Q[
                            weir * n_level_nodes_per_reach
                            + 1 : (weir + 1) * n_level_nodes_per_reach,
                            :-1,
                        ]
                    )
                    ** 2
                    * (
                        A_full_H[
                            weir * n_level_nodes_per_reach
                            + 1 : (weir + 1) * n_level_nodes_per_reach,
                            :-1,
                        ]
                        - A_full_H[
                            weir
                            * n_level_nodes_per_reach : (weir + 1)
                            * n_level_nodes_per_reach
                            - 1,
                            :-1,
                        ]
                    )
                    / dxH[
                        weir
                        * n_level_nodes_per_reach : (weir + 1)
                        * n_level_nodes_per_reach
                        - 1
                    ]
                )
                + (
                    Q_full[
                        weir * n_level_nodes_per_reach
                        + 1 : (weir + 1) * n_level_nodes_per_reach,
                        1:,
                    ]
                    - Q_full[
                        weir * n_level_nodes_per_reach
                        + 1 : (weir + 1) * n_level_nodes_per_reach,
                        :-1,
                    ]
                )
                / dt
                + g
                * (
                    self.theta
                    * A_full_Q[
                        weir * n_level_nodes_per_reach
                        + 1 : (weir + 1) * n_level_nodes_per_reach,
                        :-1,
                    ]
                    + (1 - self.theta) * A_nominal
                )
                * (
                    H_full[
                        weir * n_level_nodes_per_reach
                        + 1 : (weir + 1) * n_level_nodes_per_reach,
                        1:,
                    ]
                    - H_full[
                        weir
                        * n_level_nodes_per_reach : (weir + 1)
                        * n_level_nodes_per_reach
                        - 1,
                        1:,
                    ]
                )
                / dxH[
                    weir
                    * n_level_nodes_per_reach : (weir + 1)
                    * n_level_nodes_per_reach
                    - 1
                ]
                + g
                * (
                    self.theta
                    * P_full_Q[
                        weir * n_level_nodes_per_reach
                        + 1 : (weir + 1) * n_level_nodes_per_reach,
                        :-1,
                    ]
                    * sabs(
                        Q_full[
                            weir * n_level_nodes_per_reach
                            + 1 : (weir + 1) * n_level_nodes_per_reach,
                            :-1,
                        ]
                    )
                    / A_full_Q[
                        weir * n_level_nodes_per_reach
                        + 1 : (weir + 1) * n_level_nodes_per_reach,
                        :-1,
                    ]
                    ** 2
                    + (1 - self.theta) * P_nominal * sabs(Q_nominal) / A_nominal ** 2
                )
                * Q_full[
                    weir * n_level_nodes_per_reach
                    + 1 : (weir + 1) * n_level_nodes_per_reach,
                    1:,
                ]
                / C ** 2
                for weir in range(n_weirs)
            )
        )

        assert d.numel() == self.Q.numel() - n_weirs * n_time_steps

        # Relate boolean variables to weir flow
        e = ca.vertcat(
            *(
                self.Q[(weir + 1) * n_level_nodes_per_reach - 1, :]
                - (100 + 100 * self.delta[weir, :])
                for weir in range(n_weirs)
            )
        )

        # Constraints
        self.g = ca.veccat(c, d, e)
        self.lbg = ca.repmat(0, self.g.size1())
        self.ubg = self.lbg

        # Objective function
        self.f = ca.sum1(ca.vec(self.H ** 2))

        # Variable bounds
        lbQ = np.full(self.n_level_nodes, -np.inf)
        ubQ = np.full(self.n_level_nodes, np.inf)

        lbQ = ca.repmat(ca.DM(lbQ), 1, n_time_steps)
        ubQ = ca.repmat(ca.DM(ubQ), 1, n_time_steps)
        lbH = ca.repmat(-np.inf, self.n_level_nodes, n_time_steps)
        ubH = ca.repmat(np.inf, self.n_level_nodes, n_time_steps)

        lbdelta = np.full(n_weirs, 0)
        lbdelta = ca.repmat(ca.DM(lbdelta), 1, n_time_steps)

        ubdelta = np.full(n_weirs, 1)
        ubdelta = ca.repmat(ca.DM(ubdelta), 1, n_time_steps)

        # Optimization problem
        assert self.Q.size() == lbQ.size()
        assert self.Q.size() == ubQ.size()
        assert self.H.size() == lbH.size()
        assert self.H.size() == ubH.size()

        self.X = ca.veccat(self.Q, self.H, self.delta)
        self.lbX = ca.veccat(lbQ, lbH, lbdelta)
        self.ubX = ca.veccat(ubQ, ubH, ubdelta)

    def solve(self, lbdelta, ubdelta):
        nlp = {"f": self.f, "g": self.g, "x": self.X, "p": self.theta}
        solver = ca.nlpsol(
            "nlpsol",
            "ipopt",
            nlp,
            {
                "ipopt": {
                    "tol": 1e-3,
                    "constr_viol_tol": 1e-3,
                    "acceptable_tol": 1e-3,
                    "acceptable_constr_viol_tol": 1e-3,
                    "print_level": 0,
                    "print_timing_statistics": "no",
                    "fixed_variable_treatment": "make_constraint",
                }
            },
        )

        # Set bounds on integer variables
        w_lbX = np.array(
            self.lbX, copy=True
        )  # Make a copy to ensure this method remains re-entrant
        w_ubX = np.array(
            self.ubX, copy=True
        )  # Make a copy to ensure this method remains re-entrant

        offset = self.Q.numel() + self.H.numel()
        w_lbX[offset:] = ca.DM(lbdelta)
        w_ubX[offset:] = ca.DM(ubdelta)

        # Homotopy loop
        results = {}
        x0 = ca.repmat(0, self.X.size1())
        if self.n_theta_steps > 1:
            theta_values = np.linspace(0.0, 1.0, self.n_theta_steps)
        else:
            theta_values = [1.0]
        for theta_value in theta_values:
            solution = solver(
                lbx=w_lbX, ubx=w_ubX, lbg=self.lbg, ubg=self.ubg, p=theta_value, x0=x0
            )
            if solver.stats()["return_status"] != "Solve_Succeeded":
                raise OptimizationError(
                    "Solve failed with status {}".format(
                        solver.stats()["return_status"]
                    )
                )
            x0 = solution["x"]
            Q_res = ca.reshape(x0[: self.Q.numel()], self.Q.size1(), self.Q.size2())
            H_res = ca.reshape(
                x0[self.Q.numel() : self.Q.numel() + self.H.numel()],
                self.H.size1(),
                self.H.size2(),
            )
            d = {}
            d["objective"] = solution["f"]
            d["Q_0"] = np.array(self.Q_left).flatten()
            for i in range(self.n_level_nodes):
                d[f"Q_{i + 1}"] = np.array(
                    ca.horzcat(self.Q0[i], Q_res[i, :])
                ).flatten()
                d[f"H_{i + 1}"] = np.array(
                    ca.horzcat(self.H0[i], H_res[i, :])
                ).flatten()
            d["delta"] = x0[self.Q.numel() + self.H.numel() :]
            results[theta_value] = d
        return results


class NonLinearPruneCallback(BranchCallback):
    def __call__(self):
        lbdelta = np.array(
            [self.get_lower_bounds(name) for name in master_model.delta_names]
        )
        ubdelta = np.array(
            [self.get_upper_bounds(name) for name in master_model.delta_names]
        )
        try:
            ret = channel_model.solve(lbdelta, ubdelta)
        except OptimizationError:
            pass
        else:
            if ret[1.0]["objective"] > master_model.nonlinear_incumbent[1.0]["objective"]:
                self.prune()


class NonLinearIncumbentCallback(IncumbentCallback):
    def __call__(self):
        delta_values = np.array(
            [self.get_values(name) for name in master_model.delta_names]
        )
        lbdelta = np.zeros(len(delta_values))
        lbdelta[np.where(delta_values > 0.5)[0]] = 1.0
        ubdelta = np.ones(len(delta_values))
        ubdelta[np.where(delta_values < 0.5)[0]] = 0.0
        try:
            results = channel_model.solve(lbdelta, ubdelta)
        except OptimizationError:
            self.reject()
        else:
            objective = results[1.0]["objective"]
            if objective < master_model.nonlinear_incumbent[1.0]["objective"]:
                master_model.nonlinear_incumbent.update(results)
            else:
                self.reject()


class MasterModel:
    def __init__(self, nonlinear_model):
        self.c = cplex.Cplex()

        # Settings
        # Branch&Cut
        # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.6.2/ilog.odms.cplex.help/refcppcplex/html/branch.html
        self.c.parameters.mip.strategy.search.set(
            self.c.parameters.mip.strategy.search.values.traditional
        )
        # Find solutions with relative gap zero
        self.c.parameters.mip.pool.relgap.set(0.0)
        # Allow plenty of solutions in the pool
        self.c.parameters.mip.pool.capacity.set(1e6)
        self.c.parameters.mip.limits.populate.set(1e6)
        # Disable presolve, as we don't want to get rid of any "unused" (from the CPLEX point of view) boolean variables
        self.c.parameters.preprocessing.presolve.set(0)
        # Parallelization (disabled by default)
        self.c.parameters.threads.set(1)
        self.c.parameters.parallel.set(self.c.parameters.parallel.values.deterministic)

        # Nonlinear incumbent
        self.nonlinear_incumbent = {1.0: {"objective": np.inf}}

        # Callbacks
        self.c.register_callback(NonLinearPruneCallback)
        self.c.register_callback(NonLinearIncumbentCallback)

        # Create integer variables
        numel = nonlinear_model.delta.numel()
        self.delta_names = [f"delta[{i}]" for i in range(numel)]
        self.c.variables.add(
            lb=[0] * numel,
            ub=[1] * numel,
            types=[self.c.variables.type.binary] * numel,
            names=self.delta_names,
        )

    def solve(self):
        # Populate solution pool
        self.c.populate_solution_pool()

        # Return optimal solution
        return self.nonlinear_incumbent


# Create and solve model
channel_model = ChannelModel(n_level_nodes=10, n_time_steps=72, dt=10 * 60, n_weirs=1)
master_model = MasterModel(channel_model)
results = master_model.solve()
print(f"Solved to objective value: {results[1.0]['objective']}")
