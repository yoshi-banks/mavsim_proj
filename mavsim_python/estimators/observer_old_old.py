"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import sys
import numpy as np
from scipy import stats
sys.path.append('..')
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
import parameters.aerosonde_parameters as MAV
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
from estimators.filters import ExtendedKalmanFilterContinuousDiscrete
import parameters.estimation_parameters1 as EST

class Observer:
    def __init__(self, ts_control, initial_state = MsgState(), initial_measurements = MsgSensors()):
        # initialized estimated state message
        self.estimated_state = initial_state
        # use alpha filters to low pass filter gyros and accels
        # alpha = Ts/(Ts + tau) where tau is the LPF time constant
        self.lpf_gyro_x = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_z)
        self.lpf_accel_x = AlphaFilter(alpha=0.7, y0=initial_measurements.accel_x)
        self.lpf_accel_y = AlphaFilter(alpha=0.7, y0=initial_measurements.accel_y)
        self.lpf_accel_z = AlphaFilter(alpha=0.7, y0=initial_measurements.accel_z)
        # use alpha filters to low pass filter absolute and differential pressure
        self.lpf_abs = AlphaFilter(alpha=0.9, y0=initial_measurements.abs_pressure)
        self.lpf_diff = AlphaFilter(alpha=0.7, y0=initial_measurements.diff_pressure)
        
        # # ekf for phi and theta
        self.attitude_ekf = EkfAttitude()
        # ekf for phi and theta
        # self.attitude_ekf = ExtendedKalmanFilterContinuousDiscrete(
        #     f=self.f_attitude, 
        #     Q = EST.Q_attitude,
        #     P0= EST.P0_attitude,
        #     xhat0=EST.xhat0_attitude,
        #     Qu=EST.Qu_attitude,
        #     Ts=ts_control,
        #     N=5
        #     )

        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = EkfPosition()

        self.R_accel = EST.R_accel
        self.R_pseudo = EST.R_pseudo
        self.R_gps = EST.R_gps


    def update(self, measurement):

        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x - SENSOR.gyro_x_bias) # todo should I subtract the SENSOR bias?
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y - SENSOR.gyro_y_bias)
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z - SENSOR.gyro_z_bias)

        # invert sensor model to get altitude and airspeed
        self.estimated_state.altitude = self.lpf_abs.update(measurement.abs_pressure - 0) / (MAV.rho * MAV.gravity) # todo should I add bias?
        self.estimated_state.Va = np.sqrt(2 * self.lpf_diff.update(measurement.diff_pressure - 0) / MAV.rho) # todo should I add bias?
        # self.estimated_state.Vg = measurement.gps_Vg # todo SHOULD I ADD THIS

        # # estimate phi and theta with simple ekf
        self.attitude_ekf.update(measurement, self.estimated_state)
        # estimate phi and theta with ekf
        # u_attitude=np.array([
        #         [self.estimated_state.p],
        #         [self.estimated_state.q],
        #         [self.estimated_state.r],
        #         [self.estimated_state.Va],
        #         ])
        # xhat_attitude, P_attitude = self.attitude_ekf.propagate_model(u_attitude)
        # y_accel=np.array([
        #         [measurement.accel_x],
        #         [measurement.accel_y],
        #         [measurement.accel_z],
        #         ])
        # xhat_attitude, P_attitude = self.attitude_ekf.measurement_update(
        #     y=y_accel, 
        #     u=u_attitude,
        #     h=self.h_accel,
        #     R=self.R_accel)
        # self.estimated_state.phi = xhat_attitude.item(0)
        # self.estimated_state.theta = xhat_attitude.item(1)

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.update(measurement, self.estimated_state)

        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state
    
    def f_attitude(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            system dynamics for propagation model: xdot = f(x, u)
                x = [phi, theta].T
                u = [p, q, r, Va].T
        '''
        ##### TODO #####
        p = u.item(0)
        q = u.item(1)
        r = u.item(2)
        phi = x.item(0)
        theta = x.item(1)
        xdot = np.array([[p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)],
                         [q * np.cos(phi) - r * np.sin(phi)]])
        return xdot

    def h_accel(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            measurement model y=h(x,u) for accelerometers
                x = [phi, theta].T
                u = [p, q, r, Va].T
        '''
        ##### TODO #####
        p = u.item(0)
        q = u.item(1)
        r = u.item(2)
        Va = u.item(3)
        g = MAV.gravity
        phi = x.item(0)
        theta = x.item(1)
        y = np.array([[q * Va * np.sin(theta) + g * np.sin(theta)],
                      [r * Va * np.cos(theta) - p * Va * np.sin(theta) - g * np.cos(theta) * np.sin(phi)],
                      [-q * Va * np.cos(theta) - g * np.cos(theta) * np.cos(phi)]])
        return y


class AlphaFilter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        self.y = self.alpha * self.y + (1.-self.alpha) * u
        return self.y


class EkfAttitude:
    # implement continuous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        # todo I need to figure this section out
        self.Q = np.eye(2)*10**(-8) #np.zeros((2, 2)) todo need Q; use this to tune
        self.Q_gyro = np.eye(3)*SENSOR.gyro_sigma**2 # todo don't think I need this
        self.R_accel = np.eye(3)*SENSOR.accel_sigma**2 # todo need R_accel; calculate from sensors
        self.N = 9 # prediction steps per cycle
        self.xhat = np.array([[0], [0.05]]) # initial state: phi, theta # todo do from params
        self.P = np.eye(2)*0.1
        self.Ts = SIM.ts_control
        self.gate_threshold = stats.chi2.isf(q=0.01, df=3) #stats.chi2.isf()

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        p = state.p
        q = state.q
        r = state.r
        phi = x.item(0) # get the state from the estimate not the memory state
        theta = x.item(1)
        # G = todo what is G here?? Is it the noise?
        f_ = np.array([[p+q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)],
                      [q*np.cos(phi) - r*np.sin(phi)]])
        return f_

    def h(self, x, measurement, state):
        # measurement model y
        p = state.p
        q = state.q
        r = state.r
        Va = state.Va
        # todo check this on page 175
        # todo should I use state.theta or use x and measurements?
        h_ = np.array([[q*Va*np.sin(state.theta) + MAV.gravity*np.sin(state.theta)],
                      [r*Va*np.cos(state.theta) - p*Va*np.sin(state.theta) - MAV.gravity*np.cos(state.theta)*np.sin(state.phi)],
                      [-q*Va*np.cos(state.theta) - MAV.gravity*np.cos(state.theta)*np.cos(state.phi)]])
        return h_

    def propagate_model(self, measurement, state):
        # model propagation
        for i in range(0, self.N):

            T_p = self.Ts / self.N # todo is Tout same as Ts in algo4
            # propagate model
            self.xhat = self.xhat + T_p * self.f(self.xhat, measurement, state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, measurement, state) #jacobian()

            ##### Extra Code ######
            # compute G matrix for gyro noise
            # G = np.array([[1.,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],\
            #               [0.,np.cos(phi),-np.sin(phi)]])
            # G_d = G*self.Ts
            # Q_d = self.Q*self.Ts**2.
            # # update P with discrete time model
            # self.P = A_d @ self.P @ A_d.T + G_d@self.Q_gyro@G_d.T + Q_d
            #######################

            # convert to discrete time models
            A_d = np.eye(2) + A * T_p + A @ A * T_p**2 #  algo4
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + T_p**2 * self.Q

    def measurement_update(self, measurement, state):
        # measurement updates
        h = self.h(self.xhat, measurement, state)
        C = jacobian(self.h, self.xhat, measurement, state)
        y = np.array([[measurement.accel_x,
                       measurement.accel_y,
                       measurement.accel_z]]).T
        S_inv = np.linalg.inv(self.R_accel + C @ self.P @ C.T)
        if (y-h).T @ S_inv @ (y-h) < self.gate_threshold:
            L = self.P @ C.T @ S_inv
            tmp = np.eye(2) - L @ C
            self.P = tmp @ self.P @ tmp.T + L @ self.R_accel @ L.T
            self.xhat = self.xhat + L @ (y-h)


class EkfPosition:
    # implement continous-discrete EKF to estimate pn, pe, Vg, chi, wn, we, psi
    def __init__(self):
        self.Q = np.eye(7)*10**(-2) # todo does this work?, need to tune
        self.R_gps = np.array([[SENSOR.gps_n_sigma**2, 0, 0, 0],
                               [0, SENSOR.gps_e_sigma**2, 0, 0],
                               [0, 0, SENSOR.gps_Vg_sigma**2, 0],
                               [0, 0, 0, SENSOR.gps_course_sigma**2]])
        self.R_pseudo = np.array([[0.01, 0],
                                  [0, 0.01]])
        self.N = 4 # number of prediction step per sample
        self.Ts = SIM.ts_control
        self.xhat = np.array([[0], [0], [25], [0], [0], [0], [0]]) # todo initialize this with the starting state from params
        self.P = np.eye(7)*(0.1) # todo is this correct, how do I initialize this?
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999
        self.pseudo_threshold = stats.chi2.isf(q=0.01, df=3)
        self.gps_threshold = 100000 # don't gate GPS

    def update(self, measurement, state):
        self.propagate_model(measurement, state)
        self.measurement_update(measurement, state)
        state.north = self.xhat.item(0)
        state.east = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, measurement, state):
        # system dynamics for propagation model: xdot = f(x, u)
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        phi = state.phi
        theta = state.theta
        Va = state.Va
        q = state.q
        r = state.r

        psidot = q*np.sin(phi)/np.cos(theta) + \
                 r*np.cos(phi)/np.cos(theta)
        Vgdot = (1/Vg)*((Va*np.cos(psi)+wn)*(-Va*psidot*np.sin(psi)) +
                 (Va*np.sin(psi)+we)*(Va*psidot*np.cos(psi)))
        f_ = np.array([[Vg*np.cos(chi)],
                       [Vg*np.sin(chi)],
                       [Vgdot],
                       [MAV.gravity/Vg*np.tan(phi)],
                       [0],
                       [0],
                       [psidot]])
        return f_

    def h_gps(self, x, measurement, state):
        # measurement model for gps measurements
        pn = x.item(0)
        pe = x.item(1)
        Vg = x.item(2) # todo is this Va or Vg
        chi = x.item(3)
        h_ = np.array([[pn],
                       [pe],
                       [Vg],
                       [chi]])
        return h_

    def h_pseudo(self, x, measurement, state):
        # measurement model for wind triangle pseudo measurement
        Vg = x.item(2) # todo is this Va or Vg
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        Va = state.Va
        wn = Va * np.cos(psi) + wn - Vg*np.cos(chi)
        we = Va * np.sin(psi) + we - Vg*np.sin(chi)
        h_ = np.array([[wn],
                       [we]])
        return h_

    def propagate_model(self, measurement, state):
        # model propagation
        for i in range(0, self.N):
            T_p = self.Ts / self.N
            # propagate model
            self.xhat = self.xhat + T_p * self.f(self.xhat, measurement, state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, measurement, state) #jacobian()
            # convert to discrete time models
            A_d = np.eye(7) + A*T_p + A @ A*T_p**2 #/2
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + T_p**2*self.Q

    def measurement_update(self, measurement, state):
        # always update based on wind triangle pseudo measurement
        h = self.h_pseudo(self.xhat, measurement, state)
        C = jacobian(self.h_pseudo, self.xhat, measurement, state)
        y = np.array([[0, 0]]).T
        S_inv = np.linalg.inv(self.R_pseudo + C @ self.P @ C.T)
        if (y-h).T @ S_inv @ (y-h) < self.pseudo_threshold:
            L = self.P @ C.T @ S_inv
            tmp = np.eye(7) - L @ C
            self.P = tmp @ self.P @ tmp.T + L @ self.R_pseudo @ L.T
            self.xhat = self.xhat + L @ (y - h)

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, measurement, state)
            C = jacobian(self.h_gps, self.xhat, measurement, state)
            y_chi = wrap(measurement.gps_course, h[3, 0])
            y = np.array([[measurement.gps_n,
                           measurement.gps_e,
                           measurement.gps_Vg,
                           y_chi]]).T
            S_inv = np.linalg.inv(self.R_gps + C @ self.P @ C.T)
            if (y-h).T @ S_inv @ (y-h) < self.gps_threshold:
                L = self.P @ C.T @ S_inv
                tmp = np.eye(7) - L @ C
                self.xhat = self.xhat + L @ (y - h)
                self.P = tmp @ self.P @ tmp.T + L @ self.R_gps @ L.T

            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course


def jacobian(fun, x, measurement, state):
    # compute jacobian of fun with respect to x
    f = fun(x, measurement, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.0001  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, measurement, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J
