#!/usr/bin/env python3

import numpy as np
from scipy.linalg import solve
from time import perf_counter
try:
    from cyipopt import minimize_ipopt as minimize
except ImportError:
    print("--- cyipopt not found. Only method='v3' can be used for solve method")
from ._helpers import rref
from ._dpqcqp_cpp_wrapper import solveDPQCQP


class DPQCQP:
    def __init__(self, *args):
        Q0, c0, r0, Qs, cs, rs, A1, b1, A2, b2 = args
        self.Q0 = Q0
        self.c0 = c0
        self.r0 = r0
        self.Qs = Qs
        self.cs = cs
        self.rs = rs
        self.A1 = A1
        self.b1 = b1
        self.A2 = A2
        self.b2 = b2
        self.p = Qs.shape[0]  # Number of quadratic constraints
        # maximal number of iters for IPOPT
        self.max_iters = 25

        # Save the base A2, b2 for method 3
        self.A2_v3 = A2
        self.b2_v3 = b2

    # def setInitialPoint(self, array):
    #     assert(len(array) == self.p + self.b1.size + self.b2.size)
    #     self.y0 = np.asarray(array)

    def set_max_iters(self, max_iters):
        self.max_iters = max_iters

    def apply_bounds(self, fixed_vars_indices, bounds):
        assert fixed_vars_indices.shape == bounds.shape, "shape mismatch"
        self.fixed_vars_indices = fixed_vars_indices
        self.fixed_values = bounds
        # Start timing for rref
        t_start = perf_counter()
        # Incorporate fixed values into A2, i.e. set the first 3*fac
        # integer variabls to zero and use rref to transform A2
        # into max rank.
        A2bounds = np.zeros((bounds.shape[0], self.c0.shape[0]))
        for i, k in enumerate(fixed_vars_indices):
            A2bounds[i, k] = 1.0
        A2tmp1 = np.vstack(
            (self.A2, A2bounds))
        b2tmp1 = np.hstack((self.b2, bounds))
        _, jb = rref(A2tmp1.copy().T)
        self.A2 = A2tmp1[jb, :]
        self.b2 = b2tmp1[jb]
        # Stop timing
        t_end = perf_counter()
        self.t_rref = int((t_end - t_start) * 1e3)  # runtime in ms
        self.num_dual_vars = self.p + self.b1.size + self.b2.size

    def get_num_dual_vars(self):
        return self.num_dual_vars

    # evaluates the dual objective AND the gradient and returns
    # both
    def __ObjectiveAndGrad_v1(self, y):
        # Split the (dual) variables into alpha, Lambda, Mu
        alpha, Lambda, mu = np.split(y, [self.p, self.b1.size+self.p])
        # Calculate the helper matrices
        ralpha = self.r0 + alpha @ self.rs
        calpha = self.c0 + np.sum(alpha[:, None] * self.cs, axis=0)
        Qalpha = self.Q0 + np.sum(alpha[:, None, None] * self.Qs, axis=0)
        # Objective
        q = calpha + self.A1.T @ Lambda + self.A2.T @ mu
        z = solve(Qalpha, q, assume_a='pos',
                  check_finite=False, overwrite_a=True)
        obj = -0.5 * q.T @ z + ralpha - self.b1.T @ Lambda - self.b2.T @ mu
        # Gradient
        g_alpha = np.array(
            [0.5*z.T@Q@z - c.T@z for Q, c in zip(self.Qs, self.cs)])
        g_Lambda = -self.A1 @ z - self.b1
        g_mu = -self.A2 @ z - self.b2
        grad = np.array(
            [*g_alpha.flatten(), *g_Lambda.flatten(), *g_mu.flatten()])
        return obj, grad

    def __solve_v1(self, y0):
        # Set the bounds for the dual variables
        num_dual_vars = self.p + self.b1.size + self.b2.size
        # alpha, Lambda >= 0
        bounds = [(0, None) for _ in range(num_dual_vars)]
        # -infty < mu < infty
        bounds[-self.b2.size:] = [(None, None) for _ in range(self.b2.size)]

        # Turn the maximization problem into a minimization problem for Ipopt
        def f(y): return [-1*v for v in self.__ObjectiveAndGrad_v1(y)]

        # Set options for Ipopt (no output to the console and number of iters)
        options = {'disp': 0, 'maxiter': self.max_iters}

        # initial point
        if y0 is None:
            y0 = 0.1*np.zeros(num_dual_vars)

        # Solve with Ipopt
        t_start = perf_counter()
        res = minimize(f, x0=y0, jac=True, bounds=bounds, options=options)
        t_end = perf_counter()
        self.runtime = int((t_end - t_start)*1e3)  # runtime in ms
        self.objVal = -1 * res.fun

        # evaluates the dual objective AND the gradient and returns both
    def __ObjectiveAndGrad_v2(self, y):
        # Split the (dual) variables into alpha, Lambda, Mu
        alpha, mu = np.split(y, [self.p])
        # Calculate the helper matrices
        ralpha = self.r0 + alpha @ self.rs
        calpha = self.c0 + np.sum(alpha[:, None] * self.cs, axis=0)
        Qalpha = self.Q0 + np.sum(alpha[:, None, None] * self.Qs, axis=0)
        # Objective
        q = calpha + self.A2.T @ mu
        try:
            z = solve(Qalpha, q, assume_a='pos',
                      check_finite=False, overwrite_a=True)
        except np.linalg.LinAlgError as err:
            # Unfortunately, Qalpha is bad conditioned for ...
            z = solve(Qalpha, q, assume_a="sym",
                      check_finite=False, overwrite_a=True)
        obj = -0.5 * q.T @ z + ralpha - self.b2.T @ mu
        # Gradient
        g_alpha = np.array(
            [0.5*z.T@Q@z - c.T@z for Q, c in zip(self.Qs, self.cs)])
        g_mu = -self.A2 @ z - self.b2
        grad = np.array([*g_alpha.flatten(), *g_mu.flatten()])
        return obj, grad

    def __solve_v2(self, y0):
        # Set the bounds for the dual variables
        num_dual_vars = self.p + self.b2.size
        # -infty < mu < infty
        bounds = [(None, None) for _ in range(num_dual_vars)]
        # alpha >= 0
        bounds[:self.p] = [(0, None) for _ in range(self.p)]

        # Turn the maximization problem into a minimization problem for Ipopt
        def f(y): return [-1*v for v in self.__ObjectiveAndGrad_v2(y)]

        # Set options for Ipopt (no output to the console and number of iters)
        options = {'disp': 0, 'maxiter': self.max_iters}

        # initial point
        if y0 is None:
            y0 = np.zeros(num_dual_vars)

        # Solve with Ipopt
        t_start = perf_counter()
        res = minimize(f, x0=y0, jac=True, bounds=bounds, options=options)
        t_end = perf_counter()
        self.runtime = int((t_end - t_start)*1e3)  # runtime in ms
        self.objVal = -1 * res.fun

    def __solve_v3(self, alphas, use_nontrivial_alphas):
        if self.fixed_values is not None:
            res = solveDPQCQP(self.Q0, self.c0, self.r0, self.Qs, self.cs,
                              self.rs, self.A1, self.b1, self.A2_v3, self.b2_v3, alphas, self.fixed_vars_indices, self.fixed_values, use_nontrivial_alphas)
            self.objVal, self.runtime, self.t_rref, self.best_alpha, self.best_mu = res
        else:
            print("No fixed_values set. First run method v1 or v2 and apply the bounds.")

    def solve(self, method, alphas=None, use_nontrivial_alphas=False, initial_point=None):
        if method == "v1":
            self.__solve_v1(initial_point)
        if method == "v2":
            self.__solve_v2(initial_point)
        if method == "v3" and alphas is not None:
            self.__solve_v3(alphas, use_nontrivial_alphas)

    def getInfo(self):
        return self.objVal, self.runtime, self.t_rref

    def get_best_alpha(self):
        return self.best_alpha

    def get_best_mu(self):
        return self.best_mu
