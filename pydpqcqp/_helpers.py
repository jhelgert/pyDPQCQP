#!/usr/bin/env python3

from scipy.io import loadmat
import numpy as np
import json


def readMat(filename):
    data = loadmat(filename)
    # Number of quadratic constraints (note r = 0.0 for all quadratic constraints)
    p = (len(data.keys())-(3+8)) // 2
    Q0 = data['Q'].astype(np.float64)
    c0 = data['c'].flatten().astype(np.float64)
    r0 = 0.0
    # Quadratic constraints
    if p > 1:
        Qs = np.array([data[f"Z{i}"]
                       for i in range(1, p+1)]).astype(np.float64)
        cs = np.array([data[f"z{i}"].flatten()
                       for i in range(1, p+1)]).astype(np.float64)
        rs = np.zeros(p)
    else:
        Qs = np.array([data['Z']]).astype(np.float64)
        cs = np.array([data['z'].flatten()]).astype(np.float64)
        rs = np.array([0.0])
    # affin-linear inequality constraint
    b1 = data['b'].flatten().astype(np.float64)
    A1 = data['A'].astype(np.float64)
    # affin-linear equality constraint
    A2 = data['Aeq'].astype(np.float64)
    b2 = data['beq'].flatten().astype(np.float64)
    # bounds
    lb = data['lb'].flatten().astype(np.float64)
    ub = data['ub'].flatten().astype(np.float64)
    return Q0, c0, r0, Qs, cs, rs, A1, b1, A2, b2, lb, ub


def readJson(filename):
    # Read the json file into the data dict
    with open(filename, "r") as f:
        data = json.load(f)
    # Number of primal variables
    n = data['n']
    # Number of quadratic constraints
    p = data['p']
    # Number of affin-linear ineq. constraints
    m1 = data['m1']
    # number of affin-linear eq. constraints
    m2 = data['m2']
    # Extract the data
    Q0 = np.array(data['Q0']).astype(np.float64)
    c0 = np.array(data['c0']).astype(np.float64)
    r0 = data['r0']
    # Quadratic ineq. constraints
    Qs = np.array([data[f'Q{i+1}'] for i in range(p)]).astype(np.float64)
    cs = np.array([data[f'c{i+1}'] for i in range(p)]).astype(np.float64)
    rs = np.array([data[f'r{i+1}'] for i in range(p)]).astype(np.float64)
    # affin-linear ineq. constraint
    A1 = np.array(data['A1']).astype(np.float64)
    b1 = np.array(data['b1']).astype(np.float64)
    # affin-linear eq. constraint
    A2 = np.array(data['A2']).astype(np.float64)
    b2 = np.array(data['b2']).astype(np.float64)
    # Bounds
    lb = np.array(data['lb']).astype(np.float64)
    ub = np.array(data['ub']).astype(np.float64)
    return n, p, m1, m2, Q0, c0, r0, Qs, cs, rs, A1, b1, A2, b2, lb, ub


def rref(A, tol=1.0e-12):
    m, n = A.shape
    i, j = 0, 0
    jb = []

    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j
        k = np.argmax(np.abs(A[i:m, j])) + i
        p = np.abs(A[k, j])
        if p <= tol:
            # The column is negligible, zero it out
            A[i:m, j] = 0.0
            j += 1
        else:
            # Remember the column index
            jb.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                A[[i, k], j:n] = A[[k, i], j:n]
            # Divide the pivot row i by the pivot element A[i, j]
            A[i, j:n] = A[i, j:n] / A[i, j]
            # Subtract multiples of the pivot row from all the other rows
            for k in range(i+1, m):
                A[k, j:n] -= A[k, j] * A[i, j:n]
            i += 1
            j += 1
    # Finished
    return A, jb
