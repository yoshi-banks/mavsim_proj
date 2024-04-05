import numpy as np

def rk4(f, t, h, y):
    """
    Runge-Kutta 4th order method for numerical integration.
    
    Parameters:
        f (function): The function defining the differential equation dy/dt = f(t, y).
        t (float): The current time.
        h (float): The time step size.
        y (float): The current state.
    
    Returns:
        float: The next state after one time step.
    """
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + h/2, y + h*k2/2)
    k4 = f(t + h, y + h*k3)
    ynext = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return ynext

import numpy as np

def jacobian(fun, x, eps=0.0001):
    """
    Compute the numerical Jacobian of a function with respect to a vector.

    Parameters:
        fun (function): The function for which to compute the Jacobian.
        x (numpy.ndarray): The vector with respect to which to compute the Jacobian.
        eps (float, optional): The small perturbation used for numerical differentiation. Default is 0.0001.

    Returns:
        numpy.ndarray: The numerical Jacobian matrix.
    """
    # compute jacobian of fun with respect to x
    x = atleast_col_vector(x)
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

def atleast_col_vector(vector):
    """
    Ensure that the input vector is a column vector (2D array with one column).

    Parameters:
    vector (numpy.ndarray): The input vector to be checked.

    Returns:
    numpy.ndarray: The input vector as a column vector.
    """
    if not isinstance(vector, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    if vector.ndim > 2:
        return vector
    elif vector.ndim == 1:
        # If the input is a 1D array, reshape it into a column vector
        return vector.reshape(-1, 1)
    elif vector.ndim == 2 and vector.shape[1] == 1:
        # If the input is already a column vector, return it as is
        return vector
    elif vector.ndim == 2 and vector.shape[0] == 1:
        # If the input is a 1D array with one element, reshape it into a column vector
        return vector.reshape(-1, 1)
    else:
        # If the input is a 2D array with more than one column, raise an error
        raise ValueError("Input must be a 1D array or a 2D array with one column")

def is_column_vector(vector):
    """
    Check if the input vector is a column vector (2D array with one column).

    Parameters:
    vector (numpy.ndarray): The input vector to be checked.

    Returns:
    bool: True if the input vector is a column vector, False otherwise.
    """
    if not isinstance(vector, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    return vector.ndim == 2 and vector.shape[1] == 1

def is_square_matrix(matrix):
    """
    Check if the input matrix is a square matrix (2D array with the same number of rows and columns).

    Parameters:
    matrix (numpy.ndarray): The input matrix to be checked.

    Returns:
    bool: True if the input matrix is a square matrix, False otherwise.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]

def set_axes_equal(ax):
    # Get limits of the current axis
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])

    # Find the range of the data
    max_range = np.ptp(limits, axis=1).max() / 2.0

    # Find the midpoint of the data
    mid = np.mean(limits, axis=1)

    # Set new limits
    ax.set_xlim3d([mid[0] - max_range, mid[0] + max_range])
    ax.set_ylim3d([mid[1] - max_range, mid[1] + max_range])
    ax.set_zlim3d([mid[2] - max_range, mid[2] + max_range])
