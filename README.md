# pyDPQCQP

![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/jhelgert/pyDPQCQP)

This is a python module for quickly calculating a dual bound
to the primal problem

```
min  f(x) = 0.5 * x' * Q0 * x + c0' * x + r0
 x
s.t. g_i(x) = 0.5 * x' * Qi * x + ci' * x + r0 <= 0 for all i = 1, .., p
      h1(x) = A1*x - b1 <= 0
      h2(x) = A2*x - b2 == 0
```

- Here all the matrices Q0, ..., Qp are assumed to be symmetric positive definite, 
i.e. f and g_i are strictly convex and the primal problem is convex.
- The matrices A1 and A2 have dimensions (m1, n) and (m2, n) with m2 <= n
and A2 full rank.
- p is assumed to be small, i.e. only a few quadratic inequality constraints.

The packages requires an installation of CMake, a decent C++ compiler as well as `cyipopt`, `scipy` and
`numpy` to interface the IPOPT solver. 
All required C++ libraries (`Blaze`, `pybind11`, ..) will automatically be downloaded by CMake.

## Install

To install the package, simple clone this repo and
run

``` bash
python3 setup.py install
```
inside the repo folder.


## Example

``` python
import numpy as np
from pyDPQCQP import DPQCQP

# Primal objective:
# Q0 has shape (n, n)
# c0 has shape (n,)
# r0 is a scalar

# Primal quadratic inequality constraints:
# Qs has shape (p, n, n)
# cs has shape (p, n)
# rs has shape (p,)

# Primal affin-linear constraints:
# A1 has shape (m1, n)
# b1 has shape (m1,)
# A2 has shape (m2, n)
# b2 has shape (m2,)
dpqcqp = DPQCQP(Q0, c0, r0, Qs, cs, rs, A1, b1, A2, b2)

# Incorporate variable bounds for the primal problem, i.e.
# set x_0 = .... = x_29 = 0
dpqcqp.apply_bounds(np.arange(30), np.zeros(30))

# Optional: Set the max. number of iterations for Ipopt (default: 25)
#dpqcqp.set_max_iters(200)

# variant 1 (uses IPOPT and solves the dual Problem)
dpqcqp.solve(method="v1")
objval1, t1, t1_rref = dpqcqp.getInfo()

# variant 2 (uses IPOPT and solves a reduced dual Problem)
dpqcqp.solve(method="v2")
objval2, t2, t2_rref = dpqcqp.getInfo()

# variant 3 (performs the best)
# Note: applies the bounds again (internally) and calls a C++ implementation of rref
alphas = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.8, 0.95])
dpqcqp.solve(method="v3", alphas=alphas)
objval3, t3, t3_rref = dpqcqp.getInfo()
```
