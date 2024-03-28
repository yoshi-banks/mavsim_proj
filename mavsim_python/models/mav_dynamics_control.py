"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np
from models.mav_dynamics import MavDynamics as MavDynamicsForces
# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler


class MavDynamics(MavDynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())
        # update the message class for the true state
        self._update_true_state()


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)
        super()._rk4_step(forces_moments)
        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]

        ##### TODO #####
        # convert wind vector from world to body frame (self._wind = ?)
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)

        R = quaternion_to_rotation(np.array([e0, e1, e2, e3]))
        self._wind = R * steady_state

        # velocity vector relative to the airmass ([ur , vr, wr]= ?)
        ur = self._state.item(3) - self._wind.item(0)
        vr = self._state.item(4) - self._wind.item(1)
        wr = self._state.item(5) - self._wind.item(2)

        # compute airspeed (self._Va = ?)
        self._Va = np.sqrt(ur ** 2 + vr ** 2 + wr ** 2)

        # compute angle of attack (self._alpha = ?)
        self._alpha = np.arctan2(wr, ur)
        
        # compute sideslip angle (self._beta = ?)
        self._beta = np.arcsin(vr / self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        ##### TODO ######
        # extract states (phi, theta, psi, p, q, r)
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        # extract control inputs
        de = delta.elevator
        da = delta.aileron
        dr = delta.rudder
        dt = delta.throttle

        # compute gravitational forces ([fg_x, fg_y, fg_z])
        Fg = MAV.mass*MAV.gravity*np.array([[2*(e1*e3-e2*e0)],
                                    [2*(e2*e3 + e1*e0)],
                                    [e3**2+e0**2-e1**2-e2**2],
                                    ])


        # compute Lift and Drag coefficients (CL, CD)
        M_e = 25
        sig = lambda a: (1+np.exp(-M_e*(a-MAV.alpha0))+np.exp(M_e*(a+MAV.alpha0)))/((1+np.exp(-M_e*(a-MAV.alpha0)))*(1+np.exp(M_e*(a+MAV.alpha0))))
        cla = lambda a: (1-sig(a))*(MAV.C_L_0+MAV.C_L_alpha*a)+sig(a)*(2*np.sign(a)*np.sin(a)**2*np.cos(a))
        cda = lambda a: MAV.C_D_p + (MAV.C_L_0+MAV.C_L_alpha*a)**2/(np.pi*MAV.e*MAV.AR)

        cxa = lambda a: -(cda(a)) * np.cos(a) + (cla(a)) * np.sin(a)
        cxq = lambda a: -MAV.C_D_q * np.cos(a) + MAV.C_L_q * np.sin(a)
        cxde = lambda a: -MAV.C_D_delta_e * np.cos(a) + MAV.C_L_delta_e * np.sin(a)

        cza = lambda a: -(cda(a)) * np.sin(a) - (cla(a)) * np.cos(a)
        czq = lambda a: -MAV.C_D_q * np.sin(a) - MAV.C_L_q * np.cos(a)
        czde = lambda a: -MAV.C_D_delta_e * np.sin(a) - MAV.C_L_delta_e * np.cos(a)

        c = MAV.c/(2*self._Va)
        b = MAV.b/(2*self._Va)

        Fa = 0.5*MAV.rho*self._Va**2*MAV.S_wing*np.array([\
            [1,0,0],[0,1,0],[0,0,1]]).dot(np.array([[cxa(self._alpha)+cxq(self._alpha)*c*q+cxde(self._alpha)*de],
            [MAV.C_Y_0+MAV.C_Y_beta*self._beta+MAV.C_Y_p*b*p+MAV.C_Y_r*b*r+MAV.C_Y_delta_a*da+MAV.C_Y_delta_r*dr],
            [cza(self._alpha)+czq(self._alpha)*c*q+czde(self._alpha)*de],
            ]))

        # compute Lift and Drag Forces (F_lift, F_drag)

        # propeller thrust and torque
        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta.throttle)
        Fp = np.array([[thrust_prop], [0], [0]])
        Mp = np.array([[-torque_prop], [0], [0]])

        # compute longitudinal forces in body frame (fx, fz)

        # compute lateral forces in body frame (fy)

        # compute logitudinal torque in body frame (My)

        # compute lateral torques in body frame (Mx, Mz)

        Ma = 0.5*MAV.rho*self._Va**2*MAV.S_wing*np.array([\
            [MAV.b*(MAV.C_ell_0+MAV.C_ell_beta*self._beta+MAV.C_ell_p*b*p+MAV.C_ell_r*b*r+MAV.C_ell_delta_a*da+MAV.C_ell_delta_r*dr)],
            [MAV.c*(MAV.C_m_0+(MAV.C_m_alpha*self._alpha)+(MAV.C_m_q*c*q)+(MAV.C_m_delta_e*de))],
            [MAV.b*(MAV.C_n_0+(MAV.C_n_beta*self._beta)+(MAV.C_n_p*b*p)+(MAV.C_n_r*b*r)+(MAV.C_n_delta_a*da)+(MAV.C_n_delta_r*dr))]
            ])
        
        F = Fg + Fa + Fp 
        fx = F.item(0)
        fy = F.item(1)
        fz = F.item(2)
        M = Ma + Mp
        Mx = M.item(0)
        My = M.item(1)
        Mz = M.item(2)

        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        return forces_moments

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller
        ################################### NEW ###################################
        # see page 55 of Small Unmanned Aircraft
        ##### TODO #####
        # map delta_t throttle command(0 to 1) into motor input voltage
        # v_in = MAV.V_max * delta_t

        # Angular speed of propeller (omega_p = ?)
        # a = (MAV.rho * MAV.D_prop**5) / (2 * np.pi)**2 * MAV.C_Q0
        # b = (MAV.rho * MAV.D_prop**4) / (2 * np.pi) * MAV.C_Q1 * Va + (MAV.KQ * MAV.KV) / MAV.R_motor
        # c = (MAV.rho * MAV.D_prop**3) * MAV.C_Q2 * Va**2 - MAV.KQ / MAV.R_motor * v_in  + MAV.KQ * MAV.i0
        # omega_p = -b + np.sqrt(b**2 - 4*a*c) / (2*a)
        # omega_p = 25

        # thrust and torque due to propeller
        # J = 2 * np.pi * Va / (omega_p * MAV.D_prop)
        # CT = MAV.C_T0 + MAV.C_T1 * J + MAV.C_T2 * J**2
        # CQ = MAV.C_Q0 + MAV.C_Q1 * J + MAV.C_Q2 * J**2
        # thrust_prop = (MAV.rho * MAV.D_prop**4) / (4 * np.pi**2) * omega_p ** 2 * CT
        # torque_prop = (MAV.rho * MAV.D_prop**5) / (4 * np.pi**2) * omega_p ** 2 * CQ

        # thrust_prop1 = (MAV.rho * MAV.D_prop**4 * MAV.C_T0) / (4 * np.pi**2) * omega_p**2 \
        #     + (MAV.rho * MAV.D_prop**3 * MAV.C_T1) / (2 * np.pi) * omega_p \
        #     + (MAV.rho * MAV.D_prop**2 * MAV.C_T2 * Va**2)
        
        # torque_prop1 = (MAV.rho * MAV.D_prop**5 * MAV.C_Q0) / (4 * np.pi**2) * omega_p**2 \
        #     + (MAV.rho * MAV.D_prop * MAV.C_Q1 * Va) / (2 * np.pi) * omega_p \
        #     + (MAV.rho * MAV.D_prop**3 * MAV.C_Q2 * Va**2)
        
        ##############################################################################################

        ################################### OLD ###################################
        thrust_prop = 0.5*MAV.rho*MAV.S_prop*MAV.C_prop*((MAV.k_motor*delta_t)**2-self._Va**2)
        torque_prop = MAV.kTp*(MAV.kOmega*delta_t)**2

        ############################################################################

        return thrust_prop, torque_prop

    def _update_true_state(self):
        # rewrite this function because we now have more information
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.camera_az = 0
        self.true_state.camera_el = 0

