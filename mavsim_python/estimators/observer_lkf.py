import numpy as np
from estimators.observer_base import ObserverBase
from estimators.filter_library.src.discrete_linear_kalman_filter import DiscreteLinearKalmanFilter
import parameters.estimation_parameters_lkf as EST
from message_types.msg_sensors import MsgSensors
from scipy.linalg import expm

class ObserverLKF(ObserverBase):

    def __init__(self, t0: float, initial_measurements: MsgSensors=MsgSensors()):
        super().__init__(initial_measurements)

        x0_attitude = EST.xhat0_attitude
        P0_attitude = EST.P0_attitude        
        x0_position = EST.xhat0_position
        P0_position = EST.P0_position

        u0_attitude=np.array([
                [self.estimated_state.p],
                [self.estimated_state.q],
                [self.estimated_state.r],
                [self.estimated_state.Va],
                ])
        
        u_smooth = np.array([
            [self.estimated_state.q],
            [self.estimated_state.r],
            [self.estimated_state.Va],
            [self.estimated_state.phi],
            [self.estimated_state.theta],
            ])

        self.attitude_filter = DiscreteLinearKalmanFilter(EST.xhat0_attitude, EST.P0_attitude)

        self.position_filter = DiscreteLinearKalmanFilter(EST.xhat0_position, EST.P0_position)

        self.attitudeParams = None
        self.attitudeDynParams = {
            'F': expm(self.Afunc_attitude(x0_attitude, u0_attitude) * EST.ts_simulation),
            'G': self.Gfunc_attitude(x0_attitude, u0_attitude),
            'Q': EST.Q_attitude,
            'Qu': EST.Qu_attitude,
        }
        self.accelMeasParams = {
            'H': self.Hfunc_accel(x0_attitude, u0_attitude),
            'R': EST.R_accel,
        }

        self.positionParams = None
        self.positionDynParams = {
            'F': expm(self.Afunc_smooth(x0_position, u_smooth) * EST.ts_simulation),
            'G': self.Gfunc_smooth(x0_position, u_smooth),
            'Q': EST.Q_position,
            'Qu': EST.Qu_position,
        }
        self.pseudoMeasParams = {
            'H': self.Hfunc_pseudo(x0_position, u_smooth),
            'R': EST.R_pseudo,
        }
        self.gpsMeasParams = {
            'H': self.Hfunc_gps(x0_position, u_smooth),
            'R': EST.R_gps,
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