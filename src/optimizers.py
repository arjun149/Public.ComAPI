import numpy as np
import cvxpy as cp

def equal_weight(n):
    w = np.ones(n) / n
    return w

def min_variance(cov):
    n = cov.shape[0]
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)),
                      [cp.sum(w) == 1])
    prob.solve(solver=cp.OSQP, warm_start=True)
    return np.array(w.value).flatten()

def mean_variance(mu, cov, risk_aversion=1.0):
    n = cov.shape[0]
    w = cp.Variable(n)
    obj = -mu.T @ w + risk_aversion * cp.quad_form(w, cov)
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(w) == 1])
    prob.solve(solver=cp.OSQP, warm_start=True)
    return np.array(w.value).flatten()
