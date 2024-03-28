"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import numpy as np
import parameters.control_parameters as AP
from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from controllers.pi_control import PIControl
from controllers.pd_control_with_rate import PDControlWithRate
from controllers.tf_control import TFControl
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta


class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral-directional controllers
        self.roll_from_aileron = PDControlWithRate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = PIControl(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        # self.yaw_damper = YawDamper(
        #                 k_r=AP.yaw_damper_kr,
        #                 p_wo=AP.yaw_damper_p_wo,
        #                 Ts=ts_control)

        # self.yaw_damper = TransferFunction(
        #                 num=np.array([[AP.yaw_damper_kr, 0]]),
        #                 den=np.array([[1, AP.yaw_damper_p_wo]]),
        #                 Ts=ts_control)
        # self.yaw_damper = TFControl(
        #                 k=AP.yaw_damper_kr,
        #                 n0=0.0,
        #                 n1=1.0,
        #                 d0=AP.yaw_damper_p_wo,
        #                 d1=1,
        #                 Ts=ts_control)

        # instantiate longitudinal controllers
        self.pitch_from_elevator = PDControlWithRate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        self.altitude_from_pitch = PIControl(
                        kp=AP.altitude_kp,
                        ki=AP.altitude_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.airspeed_from_throttle = PIControl(
                        kp=AP.airspeed_throttle_kp,
                        ki=AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=1.0)
        self.commanded_state = MsgState()

    def update(self, cmd, state):
	
	#### TODO #####
        # lateral autopilot

        phi_c = self.course_from_roll.update(cmd.course_command, state.chi)  #cmd.course_command
        delta_a = -8.13462186e-09  # Trim state
        # delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p) # Controller based on chi command#
        delta_r = -1.21428507e-08
        # delta_r = self.yaw_damper.update(cmd.course_command - state.chi) # Controller based on chi command#

        # longitudinal autopilot
        h_c = cmd.altitude_command
        theta_c = np.pi/16
        # theta_c = self.altitude_from_pitch.update(h_c, state.altitude)
        delta_e = -1.24785989e-01
        # delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
        # delta_t =  3.14346798e-01 # Trim state
        delta_t = self.airspeed_from_throttle.update(cmd.airspeed_command, state.Va)

        # construct control outputs and commanded states
        delta = MsgDelta(elevator=delta_e,
                         aileron=delta_a,
                         rudder=delta_r,
                         throttle=delta_t)
        self.commanded_state.altitude = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

    def saturate(self, input, low_limit, up_limit):
        if input <= low_limit:
            output = low_limit
        elif input >= up_limit:
            output = up_limit
        else:
            output = input
        return output

# TODO replace this with the TransferFunction methods
class YawDamper:
    def __init__(self, k_r, p_wo, Ts):
        self.xi = 0
        self.Ts = Ts
        self.k_r = k_r
        self.p_wo = p_wo

    def update(self, r):
        self.xi = self.xi + self.Ts * (-self.p_wo * self.xi + self.k_r * r)
        delta_r = -self.p_wo * self.xi + self.k_r * r
        return delta_r
