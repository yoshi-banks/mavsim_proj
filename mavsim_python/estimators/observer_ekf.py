import numpy as np
from estimators.observer_filter_lib import ObserverBase
from estimators.filter_library.src.hybrid_extended_kalman_filter import HybridExtendedKalmanFilter
import parameters.estimation_parameters_ekf as EST
from message_types.msg_sensors import MsgSensors

class ObserverEKF(ObserverBase):

    def __init__(self, t0: float, initial_measurements: MsgSensors=MsgSensors()):
        super().__init__(initial_measurements)

        self.attitude_filter = HybridExtendedKalmanFilter(EST.xhat0_attitude, EST.P0_attitude, t0)

        self.position_filter = HybridExtendedKalmanFilter(EST.xhat0_position, EST.P0_position, t0)

        self.attitudeParams = None
        self.attitudeDynParams = {
            'Afunc': lambda x, u, t, params: self.Afunc_attitude(x, u),
            'Lfunc': lambda x, u, t, params: self.Lfunc_attitude(x, u),
            'Gfunc': lambda x, u, t, params: self.Gfunc_attitude(x, u),
            'f': lambda x, u, t, params: self.f_attitude(x, u),
            'fdis': lambda x, u, t, params: self.fdis_attitude(x, u),
            'Q': EST.Q_attitude,
            'Qu': EST.Qu_attitude,
        }
        self.accelMeasParams = {
            'h': lambda x, u, t, params: self.h_accel(x, u),
            'Hfunc': lambda x, u, t, params: self.Hfunc_accel(x, u),
            'Mfunc': lambda x, u, t, params: self.Mfunc_accel(x, u),
            'R': EST.R_accel,
            'gate_threshold': EST.accel_gate_threshold,
        }

        self.positionParams = None
        self.positionDynParams = {
            'Afunc': lambda x, u, t, params: self.Afunc_smooth(x, u),
            'Lfunc': lambda x, u, t, params: self.Lfunc_smooth(x, u),
            'Gfunc': lambda x, u, t, params: self.Gfunc_smooth(x, u),
            'f': lambda x, u, t, params: self.f_smooth(x, u),
            'fdis': lambda x, u, t, params: self.fdis_smooth(x, u),
            'Q': EST.Q_position,
            'Qu': EST.Qu_position,
        }
        self.pseudoMeasParams = {
            'h': lambda x, u, t, params: self.h_pseudo(x, u),
            'Hfunc': lambda x, u, t, params: self.Hfunc_pseudo(x, u),
            'Mfunc': lambda x, u, t, params: self.Mfunc_pseudo(x, u),
            'R': EST.R_pseudo,
            'gate_threshold': EST.pseudo_gate_threshold,
        }
        self.gpsMeasParams = {
            'h': lambda x, u, t, params: self.h_gps(x, u),
            'Hfunc': lambda x, u, t, params: self.Hfunc_gps(x, u),
            'Mfunc': lambda x, u, t, params: self.Mfunc_gps(x, u),
            'R': EST.R_gps,
            'gate_threshold': None,
        }

    def propagate_attitude_model(self, u: np.ndarray, t) -> tuple[np.ndarray, np.ndarray]:
        '''
        Args:
            u: p, q, r, Va

        Returns:

        '''
        self.attitude_filter.predict(u, t, self.attitudeParams, self.attitudeDynParams)
        xhat = self.attitude_filter.x_hat
        P = self.attitude_filter.P
        return xhat, P
    
    def measurement_attitude_accel_update(self, y: np.ndarray, u: np.ndarray, t) -> tuple[np.ndarray, np.ndarray]:
        '''
        Args:
            y: phi, theta
            u: p, q, r, Va
            h:
            R:
            gate_threshold:
        '''
        self.attitude_filter.update(y, u, t, self.attitudeParams, self.accelMeasParams)
        xhat = self.attitude_filter.x_hat
        P = self.attitude_filter.P
        return xhat, P
    
    def propagate_position_model(self, u: np.ndarray, t) -> tuple[np.ndarray, np.ndarray]:
        '''
        
        '''
        self.position_filter.predict(u, t, self.positionParams, self.positionDynParams)
        xhat = self.position_filter.x_hat
        P = self.position_filter.P
        return xhat, P
    
    def measurement_position_pseudo_update(self, y: np.ndarray, u: np.ndarray, t) -> tuple[np.ndarray, np.ndarray]:
        '''
        
        '''
        self.position_filter.update(y, u, t, self.positionParams, self.pseudoMeasParams)
        xhat = self.position_filter.x_hat
        P = self.position_filter.P
        return xhat, P
    
    def measurement_position_gps_update(self, y: np.ndarray, u: np.ndarray, t) -> tuple[np.ndarray, np.ndarray]:
        '''
        
        '''
        self.position_filter.update(y, u, t, self.positionParams, self.gpsMeasParams)
        xhat = self.position_filter.x_hat
        P = self.position_filter.P
        return xhat, P