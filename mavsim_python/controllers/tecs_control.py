"""
pid_control
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import numpy as np


class TECSControl:
    def __init__(self, kpE=0.0, kiE=0.0, kpB=0.0, kiB=0.0, Ts=0.01, throttle_limit=1.0, pitch_limit=1.0):
        self.kpE = kpE
        self.kiE = kiE
        self.kpB = kpB
        self.kiB = kiB
        self.Ts = Ts
        self.throttle_limit = throttle_limit
        self.pitch_limit = pitch_limit
        self.E_integrator = 0.0
        self.E_delay_1 = 0.0
        self.B_integrator = 0.0
        self.B_delay_1 = 0.0

    def update_throttle(self, E):
        
        # update the integrator using trapazoidal rule
        self.E_integrator = self.E_integrator \
                          + (self.Ts/2) * (E + self.E_delay_1)
        # PI control
        u = self.kpE * E \
            + self.kiE * self.E_integrator
        # saturate PI control at limit
        u_sat = self._saturate_throttle(u)
        # integral anti-windup
        #   adjust integrator to keep u out of saturation
        if np.abs(self.kiE) > 0.0001:
            self.E_integrator = self.E_integrator \
                              + (self.Ts / self.kiE) * (u_sat - u)
        # update the delayed variables
        self.E_delay_1 = E
        return u_sat
    
    def update_pitch(self, B):

        # update the integrator using trapazoidal rule
        self.B_integrator = self.B_integrator \
                          + (self.Ts/2) * (B + self.B_delay_1)
        # PI control
        u = self.kpB * B \
            + self.kiB * self.B_integrator
        # saturate PI control at limit
        u_sat = self._saturate_pitch(u)
        # integral anti-windup
        #   adjust integrator to keep u out of saturation
        if np.abs(self.kiB) > 0.0001:
            self.B_integrator = self.B_integrator \
                              + (self.Ts / self.kiB) * (u_sat - u)
        # update the delayed variables
        self.B_delay_1 = B
        return u_sat
    
    def _saturate_throttle(self, u):
        # saturate u at +- self.limit
        if u >= self.throttle_limit:
            u_sat = self.throttle_limit
        elif u <= -self.throttle_limit:
            u_sat = -self.throttle_limit
        else:
            u_sat = u
        return u_sat

    def _saturate_pitch(self, u):
        # saturate u at +- self.limit
        if u >= self.pitch_limit:
            u_sat = self.pitch_limit
        elif u <= -self.pitch_limit:
            u_sat = -self.pitch_limit
        else:
            u_sat = u
        return u_sat
    


