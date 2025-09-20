# pyDPQCQP

![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/jhelgert/pyDPQCQP)

This is a tiny python module for quickly calculating a dual bound to the convex primal problem

```math
\begin{align*}
\min_{x \in \mathbb{R}^n}\; f(x) &= \frac12 x^\top Q_0 x + c_0^\top x + r_0 \\
\text{s.t.} \quad g_k(x) &= \frac12 x^\top Q_k x + c_k^\top x + r_k \leq 0 \quad \forall k = 1,\ldots, p \\
      h_1(x) &= A_1 x - b_1 \leq 0 \\
      h_2(x) &= A_2 x - b_2 = 0 \\
\end{align*}
```

for a very small number $p$ of quadratic constraints like $p < 10$. Here,
- all the matrices $Q_0,\ldots,Q_p$ are symmetric positive definite
- $A_1 \in \mathbb{R}^{m_1 \times n}$, $A_2 \in \mathbb{R}^{m_2 \times n}$ and $m_2 \leq n$.
- the matrix $A_2$ has full rank.


The packages requires an installation of CMake, a decent C++ compiler, as well as `cyipopt`, `scipy` and
`numpy` to interface with the IPOPT solver. 
All required C++ libraries (`Blaze`, `pybind11`, ..) will automatically be downloaded by CMake.

## Install

To install the package, simply clone this repo and run

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
