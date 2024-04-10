import numpy as np
from estimators.filter_library.src.utils import is_square_matrix
from scipy.linalg import cholesky, sqrtm

class DiscreteUnscentedKalmanFilter:
    """
    Discrete Uncented Kalman Filter class.
    See "Optimal State Estimation" by Dan Simon, page 448.
    """

    def __init__(self, x0, P0):
        """
        Construct an instance of the DiscreteUnscentedKalmanFilter class.

        Args:
            fdis: State transition function.
            h: Measurement function.
            Q: Process noise covariance matrix.
            R: Measurement noise covariance matrix.
            x0: Initial state estimate.
            P0: Initial error covariance matrix.

            TODO:
            - Add checks for fdis, h, Q, R, x0, P0
            - Create dedicated functions for unscented transform, generate_sigma_points, and compute_weights (see rlabbe book)
            - Compare filterpy implementation
        """

        # self.fdis = fdis
        # self.h = h
        # self.Q = Q
        # self.R = R
        self.x_hat = x0
        self.P = P0

        # assert self.xdim == self.Q.shape[0]
        # assert is_square_matrix(self.Q) 
        # assert is_square_matrix(self.R) 
        assert is_square_matrix(self.P)

    @property
    def xdim(self):
        """
        Dimension of the state.

        Returns:
            xdim: Integer representing the state dimension.
        """
        return len(self.x_hat)
    
    # @property
    # def ydim(self):
    #     """
    #     Dimension of the measurement.

    #     Returns:
    #         ydim: Integer representing the measurement dimension.
    #     """
    #     return len(self.R)

    def predict(self, u, t, params, dyn_params):
        """
        Prediction step of the Discrete Unscented Kalman Filter.

        Args:
            u: Control input.
            t: Current time.
            params: Additional parameters (not used in this implementation).

        Returns:
            None
        """

        fdis = dyn_params['fdis']
        Q = dyn_params['Q']
        Qu = dyn_params['Qu']

        root = sqrtm(3 * self.P)
        x_hat_prev_arr = np.column_stack((self.x_hat + root[:self.xdim].T, self.x_hat - root[:self.xdim].T))

        x_hat_curr_arr = np.zeros((self.xdim, 2 * self.xdim))
        for i in range(2 * self.xdim):
            x_hat_prev = x_hat_prev_arr[:, i].reshape(-1, 1)
            x_hat_curr_arr[:, i] = fdis(x_hat_prev, u, t, params).flatten()

        x_hat_curr_prior = 1 / (2 * self.xdim) * np.sum(x_hat_curr_arr, axis=1)
        x_hat_curr_prior = x_hat_curr_prior.reshape(-1, 1)

        P_curr_prior = 1 / (2 * self.xdim) * (x_hat_curr_arr - x_hat_curr_prior) @ \
                (x_hat_curr_arr - x_hat_curr_prior).T + Q

        self.P = P_curr_prior
        self.x_hat = x_hat_curr_prior.reshape(-1, 1)

    def update(self, y, u, t, params, meas_params):
        """
        Update step of the Discrete Uncented Kalman Filter.

        Args:
            y: Measurement.
            u: Control input.
            t: Current time.
            params: Additional parameters (not used in this implementation).

        Returns:
            None
        """

        h = meas_params['h']
        R = meas_params['R']
        gate_threshold = meas_params['gate_threshold']

        # Equation 14.62
        root = sqrtm(3 * self.P)
        x_hat_curr_arr = np.column_stack((self.x_hat + root[:self.xdim].T, self.x_hat - root[:self.xdim].T))

        # Equation 14.63
        y_hat_curr_arr = np.zeros((y.shape[0], 2 * self.xdim))
        for i in range(2 * self.xdim):
            x_hat_curr = x_hat_curr_arr[:, i].reshape(-1, 1)
            y_hat_curr_arr[:, i] = h(x_hat_curr, u, t, params).flatten()

        # Equation 14.64
        y_hat_curr = 1 / (2 * self.xdim) * np.sum(y_hat_curr_arr, axis=1)
        y_hat_curr = y_hat_curr.reshape(-1, 1)

        # Equation 14.65
        Py = 1 / (2 * self.xdim) * (y_hat_curr_arr - y_hat_curr) @ \
            (y_hat_curr_arr - y_hat_curr).T + R

        # Equation 14.66
        x_hat_curr_prior = self.x_hat
        Pxy = 1 / (2 * self.xdim) * (x_hat_curr_arr - x_hat_curr_prior) @ \
            (y_hat_curr_arr - y_hat_curr).T

        # Equation 14.67
        P_curr_prior = self.P
        # Compute the inverse of Py
        if np.linalg.det(Py) == 0:
            Py_inv = np.linalg.inv(Py + np.eye(Py.shape[0]) * 1e-3)
        else:
            Py_inv = np.linalg.inv(Py)

        # if gate_threshold is None or (y - y_hat_curr).T @ S_inv @ (y - y_hat_curr) < gate_threshold:
        if True:
            K_curr = Pxy @ Py_inv
            x_hat_curr_posterior = x_hat_curr_prior + K_curr @ (y - y_hat_curr)
            P_curr_posterior = P_curr_prior - K_curr @ Py @ K_curr.T

            self.x_hat = x_hat_curr_posterior.reshape(-1, 1)
            self.P = P_curr_posterior
        else:
            pass