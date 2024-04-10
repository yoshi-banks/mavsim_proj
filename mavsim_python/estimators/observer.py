"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
        3/4/2024 - RWB
"""
import numpy as np
from scipy import stats
import parameters.control_parameters as CTRL
import parameters.sensor_parameters as SENSOR
import parameters.estimation_parameters_ekf as EST
import parameters.aerosonde_parameters as MAV
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
from estimators.filters import AlphaFilter, ExtendedKalmanFilterContinuousDiscrete
from tools.jacobian import jacobian

class Observer:
    def __init__(self, ts: float, initial_measurements: MsgSensors=MsgSensors()):
        self.Ts = ts  # sample rate of observer
        # initialized estimated state message
        self.estimated_state = MsgState()

        ##### TODO #####
        self.lpf_gyro_x = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=0.7, y0=initial_measurements.gyro_z)
        self.lpf_accel_x = AlphaFilter(alpha=0.7, y0=initial_measurements.accel_x)
        self.lpf_accel_y = AlphaFilter(alpha=0.7, y0=initial_measurements.accel_y)
        self.lpf_accel_z = AlphaFilter(alpha=0.7, y0=initial_measurements.accel_z)

        # use alpha filters to low pass filter absolute and differential pressure
        self.lpf_abs = AlphaFilter(alpha=0.9, y0=initial_measurements.abs_pressure)
        self.lpf_diff = AlphaFilter(alpha=0.7, y0=initial_measurements.diff_pressure)

        # ekf for phi and theta
        self.attitude_ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f_attitude, 
            Q = EST.Q_attitude,
            P0= EST.P0_attitude,
            xhat0=EST.xhat0_attitude,
            Qu=EST.Qu_attitude,
            Ts=ts,
            N=5
            )
        
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f_smooth, 
            Q=EST.Q_position,
            P0=EST.P0_position,
            xhat0=EST.xhat0_position,
            Qu=EST.Qu_position,
            Ts=ts,
            N=10
            )
        
        self.R_accel = EST.R_accel
        self.R_pseudo = EST.R_pseudo
        self.R_gps = EST.R_gps
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999

    def update(self, measurement: MsgSensors) -> MsgState:
        ##### TODO #####
        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x - SENSOR.gyro_x_bias)
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y - SENSOR.gyro_y_bias)
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z - SENSOR.gyro_z_bias)

        # invert sensor model to get altitude and airspeed
        abs_pressure = measurement.abs_pressure
        diff_pressure = measurement.diff_pressure
        self.estimated_state.altitude = self.lpf_abs.update(abs_pressure) / (MAV.rho * MAV.gravity)
        self.estimated_state.Va = np.sqrt(2 * self.lpf_diff.update(diff_pressure) / MAV.rho)

        gate_threshold = stats.chi2.isf(q=0.01, df=3) #stats.chi2.isf()

        # estimate phi and theta with ekf
        u_attitude=np.array([
                [self.estimated_state.p],
                [self.estimated_state.q],
                [self.estimated_state.r],
                [self.estimated_state.Va],
                ])
        xhat_attitude, P_attitude = self.attitude_ekf.propagate_model(u_attitude)
        y_accel=np.array([
                [measurement.accel_x],
                [measurement.accel_y],
                [measurement.accel_z],
                ])
        
        xhat_attitude, P_attitude = self.attitude_ekf.measurement_update(
            y=y_accel, 
            u=u_attitude,
            h=self.h_accel,
            R=self.R_accel,
            gate_threshold=gate_threshold)
        self.estimated_state.phi = xhat_attitude.item(0)
        self.estimated_state.theta = xhat_attitude.item(1)

        # estimate pn, pe, Vg, chi, wn, we, psi with ekf
        u_smooth = np.array([
                [self.estimated_state.q],
                [self.estimated_state.r],
                [self.estimated_state.Va],
                [self.estimated_state.phi],
                [self.estimated_state.theta],
                ])
        xhat_position, P_position=self.position_ekf.propagate_model(u_smooth)
        y_pseudo = np.array([[0.], [0.]])
        xhat_position, P_position=self.position_ekf.measurement_update(
            y=y_pseudo,
            u=u_smooth,
            h=self.h_pseudo,
            R=self.R_pseudo,
            gate_threshold=gate_threshold)
        
        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):
            y_gps = np.array([
                    [measurement.gps_n],
                    [measurement.gps_e],
                    [measurement.gps_Vg],
                    [wrap(measurement.gps_course, xhat_position.item(3))],
                    ])
            xhat_position, P_position=self.position_ekf.measurement_update(
                y=y_gps,
                u=u_smooth,
                h=self.h_gps,
                R=self.R_gps)
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

        self.estimated_state.north = xhat_position.item(0)
        self.estimated_state.east = xhat_position.item(1)
        self.estimated_state.Vg = xhat_position.item(2)
        self.estimated_state.chi = xhat_position.item(3)
        self.estimated_state.wn = xhat_position.item(4)
        self.estimated_state.we = xhat_position.item(5)
        self.estimated_state.psi = xhat_position.item(6)

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
    
    def Afunc_attitude(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            Jacobian of f(x, u) with respect to x
        '''
        func = lambda xin: self.f_attitude(xin, u)
        J = jacobian(func, x)
        return J
    
    def Gfunc_attitude(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            Jacobian of f(x, u) with respect to u
        '''
        func = lambda uin: self.f_attitude(x, uin)
        J = jacobian(func, u)
        return J
    
    def Lfunc_attitude(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            Jacobian of f(x, u) with respect to w
        '''
        return np.eye(2)

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
    
    def Hfunc_accel(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            Jacobian of h(x, u) with respect to x
        '''
        func = lambda xin: self.h_accel(xin, u)
        J = jacobian(func, x)
        return J
    
    def Mfunc_accel(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            Jacobian of h(x, u) with respect to w
        '''
        return np.eye(3)

    def f_smooth(self, x, u):
        '''
            system dynamics for propagation model: xdot = f(x, u)
                x = [pn, pe, Vg, chi, wn, we, psi].T
                u = [q, r, Va, phi, theta].T
        '''
        ##### TODO #####        
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        phi = u.item(3)
        theta = u.item(4)
        Va = u.item(2)
        q = u.item(0)
        r = u.item(1)
        psid = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)
        xdot = np.array([[Vg * np.cos(chi)], 
                         [Vg * np.sin(chi)], 
                         [((Va * np.cos(psi) + wn) * (-Va * psid * np.sin(psi)) + (Va * np.sin(psi) + we) * (Va * psid * np.cos(psi))) / Vg], 
                        #  [MAV.gravity / Vg * np.tan(phi) * np.cos(chi - psi)], 
                         [MAV.gravity / Vg * np.tan(phi)],
                         [0.], 
                         [0.], 
                         [psid]])
        return xdot
    
    def Afunc_smooth(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            Jacobian of h(x, u) with respect to x
        '''
        func = lambda xin: self.f_smooth(xin, u)
        J = jacobian(func, x)
        return J
    
    def Lfunc_smooth(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            Jacobian of h(x, u) with respect to w
        '''
        return np.eye(7)

    def h_pseudo(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            measurement model measurement model for wind triangale pseudo measurement: y=y(x, u)
                x = [pn, pe, Vg, chi, wn, we, psi].T
                u = [q, r, Va, phi, theta].T
            returns
                y = [pn, pe, Vg, chi]
        '''
        ##### TODO #####        
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        Va = u.item(2)
        y = np.array([[Va * np.cos(psi) + wn - Vg * np.cos(chi)], 
                      [Va * np.sin(psi) + we - Vg * np.sin(chi)]])
        return y
    
    def Hfunc_pseudo(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            Jacobian of h(x, u) with respect to x
        '''
        func = lambda xin: self.h_pseudo(xin, u)
        J = jacobian(func, x)
        return J
    
    def Mfunc_pseudo(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            Jacobian of h(x, u) with respect to w
        '''
        return np.eye(2)

    def h_gps(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            measurement model for gps measurements: y=y(x, u)
                x = [pn, pe, Vg, chi, wn, we, psi].T
                u = [p, q, r, Va, phi, theta].T
            returns
                y = [pn, pe, Vg, chi]
        '''
        ##### TODO #####       
        pn = x.item(0)
        pe = x.item(1)
        Vg = x.item(2)
        chi = x.item(3)  
        y = np.array([[pn], [pe], [Vg], [chi]])
        return y
    
    def Hfunc_gps(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            Jacobian of h(x, u) with respect to x
        '''
        func = lambda xin: self.h_gps(xin, u)
        J = jacobian(func, x)
        return J
    
    def Mfunc_gps(self, x: np.ndarray, u: np.ndarray)->np.ndarray:
        '''
            Jacobian of h(x, u) with respect to w
        '''
        return np.eye(4)




