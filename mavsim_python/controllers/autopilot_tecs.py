"""
autopilot block for mavsim_python - Total Energy Control System
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/14/2020 - RWB
"""
import numpy as np
import parameters.control_parameters as AP
import parameters.aerosonde_parameters as MAV
from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from controllers.pi_control import PIControl
from controllers.pd_control_with_rate import PDControlWithRate
from controllers.tecs_control import TECSControl
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta


class Autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = PDControlWithRate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = PIControl(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.yaw_damper = TransferFunction(
                        num=np.array([[AP.yaw_damper_kr, 0]]),
                        den=np.array([[1, AP.yaw_damper_p_wo]]),
                        Ts=ts_control)

        # instantiate TECS controllers
        self.tecs_controller = TECSControl(
                        AP.tecs_kpE, 
                        AP.tecs_kiE, 
                        AP.tecs_kpB, 
                        AP.tecs_kiB, 
                        ts_control, 
                        throttle_limit=1.0,
                        pitch_limit=np.radians(45))
        self.pitch_from_elevator = PDControlWithRate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))

        self.commanded_state = MsgState()

    def update(self, cmd, state):
	
	###### TODO ######
        # lateral autopilot
        chi_c = wrap(cmd.course_command, state.chi)
        phi_c = cmd.phi_feedforward + self.course_from_roll.update(chi_c, state.chi)
        phi_c = self.saturate(phi_c, -np.radians(30), np.radians(30))
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        delta_r = self.yaw_damper.update(state.r)

        # longitudinal TECS autopilot
        K_error = 0.5 * MAV.mass * (cmd.airspeed_command**2 - state.Va**2)
        h_error = self.saturate(cmd.altitude_command - state.altitude, -AP.h_error_max, AP.h_error_max)
        # h_error = self.saturate(state.altitude - cmd.altitude_command, -AP.h_error_max, AP.h_error_max)
        U_error = MAV.mass * MAV.gravity * h_error
        E = U_error + K_error
        B = U_error - K_error
        delta_t = self.tecs_controller.update_throttle(E)
        delta_t = self.saturate(delta_t, 0.0, 1.0)
        theta_c = self.tecs_controller.update_pitch(B)
        delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.q)

        # construct output and commanded states
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
