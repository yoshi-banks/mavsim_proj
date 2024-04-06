import numpy as np
# from src.utils import is_column_vector, is_square_matrix, rk4
from estimators.filter_library.src.utils import is_column_vector, is_square_matrix, rk4

class HybridExtendedKalmanFilter:   
    """
    Hybrid Extended Kalman Filter class.
    See "Optimal State Estimation" by Dan Simon, page 406.
    """

    def __init__(self, x0, P0, t0):
        """
        Construct an instance of the HybridExtendedKalmanFilter class.

        Args:
            f: State transition function.
            h: Measurement function.
            Q: Process noise covariance matrix.
            R: Measurement noise covariance matrix.
            x0: Initial state estimate.
            P0: Initial error covariance matrix.
            t0: Initial time.
            Afunc: Function to compute A matrix (state transition Jacobian).
            Lfunc: Function to compute L matrix (process noise Jacobian).
            Hfunc: Function to compute H matrix (measurement Jacobian).
            Mfunc: Function to compute M matrix (measurement noise Jacobian).
        """
        # add checks for F, G, H, Q, R, x0, P0
        # assert isinstance(R, np.ndarray)
        assert isinstance(x0, np.ndarray)
        assert isinstance(P0, np.ndarray)
        assert is_column_vector(x0)

        self._xdim = x0.shape[0]  # Dimension of state vector
        # self._udim = 
        # self._ydim = R.shape[0]

        # assert is_square_matrix(Q) and is_square_matrix(R) and is_square_matrix(P0)
        # assert Q.shape[0] == self._xdim and Q.shape[1] == self._xdim
        # assert R.shape[0] == self._ydim and R.shape[1] == self._ydim

        # self.f = f
        # self.h = h
        # self.Q = Q
        # self.Qu = Qu
        # self.R = R
        self.x_hat_minus = x0
        self.P_minus = P0
        self.x_hat_plus = x0
        self.P_plus = P0
        self.x_hat = x0
        self.P = P0
        self.t_minus = t0

        # self.Afunc = Afunc
        # self.Hfunc = Hfunc
        # self.Mfunc = Mfunc
        # self.Lfunc = Lfunc

        # TODO add checks that these functions return the right sized output


    def predict(self, u, t, params, dyn_params):
        """
        Prediction step of the Hybrid EKF.

        Args:
            u: Control input.
            t: Current time.
            params: Additional parameters (not used in this implementation).
            dyn_funcs: Dictionary of dynamics functions.
                'f': State transition function.
                'Gfunc': Function to compute G matrix (control input Jacobian).
                'Afunc': State transition function.
                'Lfunc': Function to compute L matrix (process noise Jacobian).
                'Q': Process noise covariance matrix.
                'Qu': Control input noise covariance matrix.
                
        Returns:
            None
        """
        x = self.x_hat
        P = self.P

        A = dyn_params['Afunc'](x, u, t, params)
        L = dyn_params['Lfunc'](x, u, t, params)
        G = dyn_params['Gfunc'](x, u, t, params)
        f = dyn_params['f']
        Q = dyn_params['Q']
        Qu = dyn_params['Qu']

        # forward dynamics using Runge-Kutta 4th order method
        dt = t - self.t_minus
        self.t_minus = t
        simfunc = lambda tin, xin: f(xin, u, tin, params)
        self.x_hat_minus = rk4(simfunc, t, dt, x)

        # forward state covariance using Runge-Kutta 4th order method
        simfunc = lambda tin, Pin: A @ Pin + Pin @ A.T + L @ Q @ L.T #+ G @ Qu @ G.T
        self.P_minus = rk4(simfunc, t, dt, P)

        self.x_hat = self.x_hat_minus
        self.P = self.P_minus
    
    def update(self, y, u, t, params, meas_params):
        """
        Update step of the Hybrid EKF.

        Args:
            z: Measurement.
            u: Control input.
            t: Current time.
            params: Additional parameters (not used in this implementation).
            meas_funcs: Dictionary of measurement functions.
                'h': Measurement function.
                'Hfunc': Function to compute H matrix (measurement Jacobian).
                'Mfunc': Function to compute M matrix (measurement noise Jacobian).

        Returns:
            None
        """
        x = self.x_hat
        P = self.P

        H = meas_params['Hfunc'](x, u, t, params)
        M = meas_params['Mfunc'](x, u, t, params)
        h = meas_params['h']
        R = meas_params['R']
        gate_threshold = meas_params['gate_threshold']

        S_inv = np.linalg.inv(H @ P @ H.T + M @ R @ M.T)
        yhat = h(x, u, t, params)

        if gate_threshold is None or (y - yhat).T @ S_inv @ (y - yhat) < gate_threshold:
            K = P @ H.T @ S_inv
            IKH = np.eye(P.shape[0]) - K @ H
            P = IKH @ P @ IKH.T + K @ M @ R @ M.T @ K.T
            x = x + K @ (y - yhat)

        self.x_hat_plus = x
        self.P_plus = P
        self.x_hat = x
        self.P = P
        
