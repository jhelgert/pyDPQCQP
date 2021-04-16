# pyDPQCQP

This is a simple python module for quickly calculating a dual bound
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
`numpy` to interface the IPOPT solver. In additon, it requires the `cplex` and `gurobipy` python packages. 
All required C++ libraries (`Blaze`, `pybind11`, ..) will automatically be downloaded by CMake.

To install the package, simple clone this repo and
run

```
python3 setup.py install
```
inside the repo folder.
