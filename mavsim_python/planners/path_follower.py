import numpy as np
from math import sin, cos
import parameters.aerosonde_parameters as MAV
from message_types.msg_state import MsgState
from message_types.msg_path import MsgPath
from message_types.msg_autopilot import MsgAutopilot
from tools.wrap import wrap


class PathFollower:
    def __init__(self):
        ##### TODO #####
        self.chi_inf = np.radians(60)  # approach angle for large distance from straight-line path
        self.k_path = 0.05  # path gain for straight-line path following
        # self.k_orbit = 10.0  # path gain for orbit following
        self.k_orbit = 20.0  # path gain for orbit following
        self.gravity = MAV.gravity
        self.autopilot_commands = MsgAutopilot()  # message sent to autopilot

    def update(self, 
               path: MsgPath, 
               state: MsgState)->MsgAutopilot:
        if path.type == 'line':
            self._follow_straight_line(path, state)
        elif path.type == 'orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, 
                              path: MsgPath, 
                              state: MsgState):
        ##### TODO #####
        #airspeed command
        self.autopilot_commands.airspeed_command = 25.0

        # calculate epy to pass through equation 10.8
        q = path.line_direction
        r = path.line_origin
        chi_q = np.arctan2(q.item(1), q.item(0))
        chi_q = wrap(chi_q, state.chi)
        Rpi = np.array([[np.cos(chi_q), np.sin(chi_q), 0],
                        [-np.sin(chi_q), np.cos(chi_q), 0],
                        [0, 0, 1]])
        p = np.array([[state.north, state.east, state.altitude]]).T
        ep = Rpi @ (p - r)
        epy = ep.item(1)

        # course command
        self.autopilot_commands.course_command = \
            chi_q - self.chi_inf * (2 / np.pi) * np.arctan(self.k_path * epy) # equation 10.8

        # altitude command
        k = np.array([[0, 0, 1]])
        n = (np.cross(k, q.T) / np.linalg.norm(np.cross(k, q.T))).T
        eip = p - r
        s = eip - (eip.T @ n) * n
        s = s.reshape(3, 1)
        rd = r.item(2)
        sn = s.item(0)
        se = s.item(1)
        qn = q.item(0)
        qe = q.item(1)
        qd = q.item(2)

        self.autopilot_commands.altitude_command = -rd - np.sqrt(sn**2 + se**2) \
                                                    * (qd / np.sqrt(qn**2 + qe**2))

        # feedforward roll angle for straight line is zero, why?
        self.autopilot_commands.phi_feedforward = 0

    def _follow_orbit(self, 
                      path: MsgPath, 
                      state: MsgState):
        ##### TODO #####
        p = np.array([[state.north, state.east, state.altitude]]).T
        Vg = state.Vg
        chi = state.chi
        psi = state.psi
        if path.orbit_direction == 'CW':
            direction = 1.0
        elif path.orbit_direction == 'CCW':
            direction = -1.0
        else:
            direction = path.orbit_direction

        # airspeed command
        self.autopilot_commands.airspeed_command = 25.0

        # course command
        c = path.orbit_center
        rho = path.orbit_radius
        d = np.sqrt((p.item(0) - c.item(0))**2 + (p.item(1) - c.item(1))**2)
        varphi = np.arctan2(p.item(1) - c.item(1), p.item(0) - c.item(0))
        varphi = wrap(varphi, state.phi)
        chi_c = varphi + direction * (np.pi / 2 + np.arctan(self.k_orbit * (d - rho) / rho))
        orbit_error = 0 # TODO what to do with this?

        self.autopilot_commands.course_command = chi_c

        # altitude command
        self.autopilot_commands.altitude_command = -c.item(2)
        
        # roll feedforward command
        if orbit_error < 10:
            self.autopilot_commands.phi_feedforward = \
                direction * np.arctan(Vg**2 / (self.gravity * rho * np.cos(chi - psi)))
        else:
            self.autopilot_commands.phi_feedforward = \
                direction * np.arctan(Vg**2 / (self.gravity * rho * np.cos(chi - psi)))




