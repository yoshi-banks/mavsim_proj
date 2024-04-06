import numpy as np
from estimators.observer_filter_lib import ObserverFilterLib
from estimators.filter_library.src.discrete_unscented_kalman_filter import DiscreteUnscentedKalmanFilter
import parameters.estimation_parameters1 as EST
from message_types.msg_sensors import MsgSensors

class ObserverEKF(ObserverFilterLib):

    def __init__(self, t0: float, initial_measurements: MsgSensors=MsgSensors()):
        super().__init__(initial_measurements)

        self.attitude_filter = 

        self.position_filter = 
