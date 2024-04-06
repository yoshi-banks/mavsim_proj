"""
Jacobian - take the numerical Jacobian of fun with respect to x
"""
import numpy as np

def Jacobian(fun, x, eps=0.0001):
    return jacobian(fun, x, eps)

def jacobian(fun, x, eps=0.0001):
    # compute jacobian of fun with respect to x
    f = fun(x)
    m = f.shape[0]
    n = x.shape[0]
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J

def partial_vector_derivative(fun, x, eps=0.001):
    assert x.shape[1] == 1, "x must be a column vector"
    return jacobian(fun, x, eps)


