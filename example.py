# Example code illustrating homotopy - CPLEX coupling for global mixed-integer non-linear optimization
#
# Copyright (C) 2017 - 2019 Jorn Baayen
# Copyright (C) 2019 KISTERS AG
# Copyright (C) 2019 IBM
#
# Authors:  Jorn Baayen, Jakub Marecek
#
# Note that this is a rather naive implementation of the continuation algorithm,
# and a naive discretization of the Saint-Venant equations exhibiting several known
# practical issues (while still, however, satisfying the mathematical requirements).
# For real-life applications, it is suggested to use an enterprise implementation,
# such as KISTERS Real Time Optimization (RTO).

import time, sys
import casadi as ca
import numpy as np
import multiprocessing
import threading
import pickle
import pprint

import cplex
from cplex.exceptions import CplexError
from cplex.callbacks import IncumbentCallback, BranchCallback


class OptimizationError(Exception):
    pass


class ChannelModel:
    def __init__(
        self, n_weirs, n_level_nodes, horizon_length, hydraulic_dt, control_dt
    ):
        assert n_level_nodes % n_weirs == 0
        assert horizon_length % hydraulic_dt == 0
        assert horizon_length % control_dt == 0

        # These parameters correspond to Table 1
        n_hydraulic_time_steps = horizon_length // hydraulic_dt
        n_control_time_steps = horizon_length // control_dt
        control_times = np.arange(
            0, (n_control_time_steps + 1) * control_dt, control_dt
        )
        self.hydraulic_times = np.arange(
            0, (n_hydraulic_time_steps + 1) * hydraulic_dt, hydraulic_dt
        )
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
        alpha = 10
        min_depth = 1e-2

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
        self.Q0 = np.full(n_level_nodes_per_reach, 100.0)
        self.H0 = np.full(n_level_nodes_per_reach, 0.0)
        for i in range(1, n_level_nodes_per_reach):
            A = w / 2.0 * (self.H0[i - 1] - H_b[i - 1] + self.H0[i] - H_b[i])
            P = w + self.H0[i - 1] - H_b[i - 1] + self.H0[i] - H_b[i]
            self.H0[i] = (
                self.H0[i - 1]
                - dx * (P / A ** 3) * sabs(self.Q0[i]) * self.Q0[i] / C ** 2
            )
        self.Q0 = np.tile(self.Q0, n_weirs)
        self.H0 = np.tile(self.H0, n_weirs)

        # Symbols
        Q = ca.MX.sym("Q", self.n_level_nodes, n_hydraulic_time_steps)
        H = ca.MX.sym("H", self.n_level_nodes, n_hydraulic_time_steps)
        delta = ca.MX.sym("delta", n_weirs, n_control_time_steps)
        theta = ca.MX.sym("theta")

        # Left boundary condition
        self.Q_left = np.full(n_hydraulic_time_steps + 1, 100)
        self.Q_left[
            n_hydraulic_time_steps // 3 : 2 * n_hydraulic_time_steps // 3
        ] = 300.0
        self.Q_left = ca.DM(self.Q_left).T

        # Hydraulic constraints
        Q_full = ca.vertcat(self.Q_left, ca.horzcat(self.Q0, Q))
        H_full = ca.horzcat(self.H0, H)
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
            theta * (A_full_H[:, 1:] - A_full_H[:, :-1])
            + (1 - theta) * w * (depth_full[:, 1:] - depth_full[:, :-1])
        ) / hydraulic_dt + (Q_full[1:, 1:] - Q_full[:-1, 1:]) / dxQ
        d = ca.vertcat(
            *(
                theta
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
                / hydraulic_dt
                + g
                * (
                    theta
                    * A_full_Q[
                        weir * n_level_nodes_per_reach
                        + 1 : (weir + 1) * n_level_nodes_per_reach,
                        :-1,
                    ]
                    + (1 - theta) * A_nominal
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
                    theta
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
                    + (1 - theta) * P_nominal * sabs(Q_nominal) / A_nominal ** 2
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

        assert d.numel() == Q.numel() - n_weirs * n_hydraulic_time_steps

        # Relate boolean variables to weir flow
        delta_full = ca.horzcat(ca.repmat(0, n_weirs, 1), delta)
        delta_interpolated = ca.transpose(
            ca.interp1d(
                control_times,
                ca.transpose(delta_full),
                self.hydraulic_times,
                "linear",
                True,
            )
        )
        e = ca.vertcat(
            *(
                Q[(weir + 1) * n_level_nodes_per_reach - 1, :]
                - (100 + 100 * delta_interpolated[weir, 1:])
                for weir in range(n_weirs)
            )
        )

        # Constraints
        g = ca.veccat(c, d, e)
        self.lbg = np.full(g.size1(), 0)
        self.ubg = self.lbg

        # Objective function
        f = ca.sum1(ca.vec(H ** 2))

        # Variable bounds
        lbQ = np.full(self.n_level_nodes, -np.inf)
        ubQ = np.full(self.n_level_nodes, np.inf)

        lbQ = ca.repmat(ca.DM(lbQ), 1, n_hydraulic_time_steps)
        ubQ = ca.repmat(ca.DM(ubQ), 1, n_hydraulic_time_steps)
        lbH = ca.repmat(-np.inf, self.n_level_nodes, n_hydraulic_time_steps)
        ubH = ca.repmat(np.inf, self.n_level_nodes, n_hydraulic_time_steps)

        lbdelta = np.full(n_weirs, 0)
        lbdelta = ca.repmat(ca.DM(lbdelta), 1, n_control_time_steps)

        ubdelta = np.full(n_weirs, 1)
        ubdelta = ca.repmat(ca.DM(ubdelta), 1, n_control_time_steps)

        # Optimization problem
        assert Q.size() == lbQ.size()
        assert Q.size() == ubQ.size()
        assert H.size() == lbH.size()
        assert H.size() == ubH.size()

        X = ca.veccat(Q, H, delta)

        self.lbX = np.array(ca.veccat(lbQ, lbH, lbdelta))
        self.ubX = np.array(ca.veccat(ubQ, ubH, ubdelta))

        nlp = {"f": f, "g": g, "x": X, "p": theta}
        self.solver = ca.nlpsol(
            "nlpsol",
            "ipopt",
            nlp,
            {
                "expand": True,
                "ipopt": {
                    "tol": 1e-3,
                    "constr_viol_tol": 1e-3,
                    "acceptable_tol": 1e-3,
                    "acceptable_constr_viol_tol": 1e-3,
                    "print_level": 0,
                    "print_timing_statistics": "no",
                    "fixed_variable_treatment": "make_constraint",
                    "linear_system_scaling": "none",
                    "nlp_scaling_method": "none",
                    # "linear_solver": "ma57",  # Required for parallel mode, as mumps is not thread-safe
                    # "ma57_automatic_scaling": "no",
                    # The following two settings are there to limit time spent
                    # on solutions with extensive drying of the canal, which
                    # are expensive to compute.  This is one of the hacks
                    # we have to insert to make this naive implementation run.
                    # Such hacks are not present in RTO.
                    "max_iter": 300,
                    "max_cpu_time": 6,
                },
            },
        )

        self.X_numel = X.numel()

        self.Q_size1 = Q.size1()
        self.Q_size2 = Q.size2()
        self.Q_numel = Q.numel()

        self.H_size1 = H.size1()
        self.H_size2 = H.size2()
        self.H_numel = H.numel()

        self.delta_numel = delta.numel()

    def solve(self, lbdelta, ubdelta):
        # Set bounds on integer variables
        w_lbX = np.array(
            self.lbX, copy=True
        )  # Make a copy to ensure this method remains re-entrant
        w_ubX = np.array(
            self.ubX, copy=True
        )  # Make a copy to ensure this method remains re-entrant

        offset = self.Q_numel + self.H_numel
        w_lbX[offset:] = ca.DM(lbdelta)
        w_ubX[offset:] = ca.DM(ubdelta)

        # Homotopy loop
        results = {}
        x0 = ca.repmat(0, self.X_numel)
        if self.n_theta_steps > 1:
            theta_values = np.linspace(0.0, 1.0, self.n_theta_steps)
        else:
            theta_values = [1.0]
        for theta_value in theta_values:
            solution = self.solver(
                lbx=w_lbX, ubx=w_ubX, lbg=self.lbg, ubg=self.ubg, p=theta_value, x0=x0
            )
            if self.solver.stats()["return_status"] != "Solve_Succeeded":
                raise OptimizationError(
                    "Solve failed with status {}".format(
                        self.solver.stats()["return_status"]
                    )
                )
            x0 = solution["x"]
            Q_res = ca.reshape(x0[: self.Q_numel], self.Q_size1, self.Q_size2)
            H_res = ca.reshape(
                x0[self.Q_numel : self.Q_numel + self.H_numel],
                self.H_size1,
                self.H_size2,
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
            d["delta"] = x0[self.Q_numel + self.H_numel :]
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
            res = master_model.pool.apply_async(channel_model_solve, (lbdelta, ubdelta))
            results = res.get()
        except KeyboardInterrupt:
            sys.exit(-1)
        except OptimizationError:
            # The following two settings are there to limit time spent
            # on solutions with extensive drying of the canal, which
            # are expensive to compute.  This is one of the hacks
            # we have to insert to make this naive implementation run.
            # Such hacks are not present in RTO.
            self.prune()
            print(f"pruned branch whose relaxation took too long to solve")
        else:
            if (
                results[1.0]["objective"]
                > master_model.nonlinear_incumbent[1.0]["objective"]
            ):
                self.prune()
                print(
                    f"pruned branch with relaxation objective value {results[1.0]['objective']}"
                )


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
            res = master_model.pool.apply_async(channel_model_solve, (lbdelta, ubdelta))
            results = res.get()
        except KeyboardInterrupt:
            sys.exit(-1)
        except OptimizationError:
            self.reject()
        else:
            objective = results[1.0]["objective"]
            with master_model.nonlinear_incumbent_lock:
                # if True:
                if objective < master_model.nonlinear_incumbent[1.0]["objective"]:
                    master_model.nonlinear_incumbent.update(results)
                    print(f"updated incumbent with objective value {objective}")
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
        # Parallelization
        self.c.parameters.threads.set(multiprocessing.cpu_count())
        self.c.parameters.parallel.set(self.c.parameters.parallel.values.opportunistic)

        # Nonlinear incumbent
        self.nonlinear_incumbent = {1.0: {"objective": np.inf}}
        self.nonlinear_incumbent_lock = threading.Lock()

        # Process pool
        def pool_initializer(channel_model_pickle):
            global channel_model
            channel_model = pickle.loads(channel_model_pickle)

        self.pool = multiprocessing.Pool(
            initializer=pool_initializer,
            initargs=(pickle.dumps(channel_model),),
            processes=multiprocessing.cpu_count(),
        )

        # Callbacks
        self.c.register_callback(NonLinearPruneCallback)
        self.c.register_callback(NonLinearIncumbentCallback)

        # Create integer variables
        numel = nonlinear_model.delta_numel
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
n_weirs = 1
channel_model = ChannelModel(
    n_level_nodes=10 * n_weirs,
    horizon_length=24 * 60 * 60,
    hydraulic_dt=10 * 60,
    control_dt=4 * 60 * 60,
    n_weirs=n_weirs,
)


def channel_model_solve(lbdelta, ubdelta):
    return channel_model.solve(lbdelta, ubdelta)


master_model = MasterModel(channel_model)

results = master_model.solve()
times = channel_model.hydraulic_times  # For compatibility with plotting scripts

pprint.pprint(results)
with open("results.p", "wb") as f:
    pickle.dump(results, f)
print(f"Solved to objective value: {results[1.0]['objective']}")
