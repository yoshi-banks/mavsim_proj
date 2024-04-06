import numpy as np
from estimators.observer_filter_lib import ObserverFilterLib
from estimators.filter_library.src.hybrid_extended_kalman_filter import HybridExtendedKalmanFilter
import parameters.estimation_parameters1 as EST
from message_types.msg_sensors import MsgSensors

class ObserverEKF(ObserverFilterLib):

    def __init__(self, t0: float, initial_measurements: MsgSensors=MsgSensors()):
        super().__init__(initial_measurements)

        self.attitude_filter = HybridExtendedKalmanFilter(EST.xhat0_attitude, EST.P0_attitude, t0)

        self.position_filter = HybridExtendedKalmanFilter(EST.xhat0_position, EST.P0_position, t0)
