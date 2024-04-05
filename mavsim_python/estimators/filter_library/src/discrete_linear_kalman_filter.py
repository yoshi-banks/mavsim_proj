import numpy as np
from src.utils import is_column_vector

class DiscreteLinearKalmanFilter:
    """
    Discrete Linear Kalman Filter class.
    See "Optimal State Estimation" by Dan Simon, page 128.
    """

    def __init__(self, F, G, H, Q, R, x0, P0):
        """
        Construct an instance of the DiscreteLinearKalmanFilter class.

        Args:
            F: State transition matrix.
            G: Control input matrix.
            H: Measurement matrix.
            Q: Process noise covariance matrix.
            R: Measurement noise covariance matrix.
            x0: Initial state estimate.
            P0: Initial error covariance matrix.
        """
        # add checks for F, G, H, Q, R, x0, P0
        if not isinstance(F, np.ndarray) or not isinstance(G, np.ndarray) \
        or not isinstance(H, np.ndarray) or not isinstance(Q, np.ndarray) \
        or not isinstance(R, np.ndarray) or not isinstance(x0, np.ndarray) \
        or not isinstance(P0, np.ndarray):
            raise TypeError("Matrices and vectors must be NumPy arrays")

        if not is_column_vector(x0):
            raise ValueError("Initial state estimate must be a column vector")
        
        self._xdim = x0.shape[0]  # Dimension of state vector
        self._udim = G.shape[1] if len(G.shape) > 1 else 1  # Dimension of control input vector
        self._ydim = H.shape[0]  # Dimension of measurement vector

        # Check dimensions of matrices
        if F.shape != (self._xdim, self._xdim):
            raise ValueError("F should be xdim by xdim")
        if G.shape != (self._xdim, self._udim):
            raise ValueError("G should be xdim by udim")
        if H.shape != (self._ydim, self._xdim):
            raise ValueError("H should be ydim by xdim")
        if Q.shape != (self._xdim, self._xdim):
            raise ValueError("Q should be xdim by xdim")
        if R.shape != (self._ydim, self._ydim):
            raise ValueError("R should be ydim by ydim")
        if x0.shape != (self._xdim, 1):
            raise ValueError("x0 should be xdim by 1")
        if P0.shape != (self._xdim, self._xdim):
            raise ValueError("P0 should be xdim by xdim")

        self.F = F
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R
        self.x_hat_minus = x0
        self.P_minus = P0
        self.x_hat_plus = x0
        self.P_plus = P0
        self.x_hat = x0
        self.P = P0

    def predict(self, u):
        """
        Prediction step of the Kalman filter.

        Args:
            u: Control input.

        Returns:
            None
        """
        assert u.shape[0] == self._udim
        assert is_column_vector(u)

        x = self.x_hat_plus
        P = self.P_plus

        F = self.F
        G = self.G

        self.P_minus = F @ P @ F.T + self.Q
        self.x_hat_minus = F @ x + G @ u
        self.x_hat = self.x_hat_minus
        self.P = self.P_minus

    def update(self, z, u, params):
        """
        Update step of the Kalman filter.

        Args:
            z: Measurement.
            u: Control input.
            params: Additional parameters (not used in this implementation).

        Returns:
            None
        """
        assert z.shape[0] == self._ydim
        assert is_column_vector(z)
        assert u.shape[0] == self._udim
        assert is_column_vector(u)

        x = self.x_hat_minus
        P = self.P_minus

        H = self.H

        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + self.R)
        self.x_hat_plus = x + K @ (z - H @ x)
        IKH = np.eye(P.shape[0]) - K @ H
        self.P_plus = IKH @ P @ IKH.T + K @ self.R @ K.T
        self.x_hat = self.x_hat_plus
        self.P = self.P_plus
