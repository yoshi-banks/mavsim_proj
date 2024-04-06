"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/4/2019 - RWB
        3/6/2024 - RWB
"""
import numpy as np
from scipy import stats
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
import parameters.aerosonde_parameters as MAV
import parameters.estimation_parameters as EST
from tools.rotations import euler_to_rotation
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
from estimators.filters import AlphaFilter, ExtendedKalmanFilterContinuousDiscrete
from tools.linalg import S, cross


class Observer:
    # def __init__(self, ts):
    def __init__(self, ts, x):
        # initialized estimated state message
        ##### TODO #####        
        Q = x.reshape(14, 14)
        self.ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f, 
            Q = Q,
            P0 = EST.P0,
            xhat0 = EST.xhat0, 
            Qu = EST.Qu,
            Ts=ts,
            N=10
            )
        
        self.R_analog = EST.R_analog
        self.R_gps = EST.R_gps
        self.R_pseudo = EST.R_pseudo
        initial_measurements = MsgSensors()

        self.lpf_gyro_x = AlphaFilter(alpha=0.3, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=0.3, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=0.3, y0=initial_measurements.gyro_z)

        self.analog_threshold = stats.chi2.isf(q=0.01, df=3)
        self.pseudo_threshold = stats.chi2.isf(q=0.01, df=2)

        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999
        self.estimated_state = MsgState()
        self.elapsed_time = 0

        # state
        # pn, pe, pd, u, v, w, phi, theta, psi, bx, by, bz, wn, we
        # input
        # gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z

    def update(self, measurement):

        # system input
        u = np.array([[
            measurement.gyro_x, 
            measurement.gyro_y, 
            measurement.gyro_z,
            measurement.accel_x, 
            measurement.accel_y, 
            measurement.accel_z,
            ]]).T
        
        xhat, P = self.ekf.propagate_model(u)

        # update with analog measurement
        y_analog = np.array([
            [measurement.abs_pressure],
            [measurement.diff_pressure],
            [0.0], # sideslip pseudo measurement
            ])
        xhat, P = self.ekf.measurement_update(
            y=y_analog, 
            u=u,
            h=self.h_analog,
            R=self.R_analog,
            gate_threshold=self.analog_threshold)
        
        # update with wind triangle pseudo measurement
        y_pseudo = np.array([
            [0.],
            [0.], 
            ])
        xhat, P = self.ekf.measurement_update(
            y=y_pseudo, 
            u=u,
            h=self.h_pseudo,
            R=self.R_pseudo,
            gate_threshold=self.pseudo_threshold)
        
        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):
            state = to_MsgState(xhat) 
                # need to do this to get the current chi to wrap meaurement
            y_chi = wrap(measurement.gps_course, state.chi)
            y_gps = np.array([
                [measurement.gps_n], 
                [measurement.gps_e], 
                [measurement.gps_Vg], 
                [y_chi]])
            xhat, P = self.ekf.measurement_update(
                y=y_gps, 
                u=u,
                h=self.h_gps,
                R=self.R_gps)
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

        # convert internal xhat to MsgState format
        self.estimated_state = to_MsgState(xhat)
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x - self.estimated_state.bx)
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y - self.estimated_state.by)
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z - self.estimated_state.bz)
        self.elapsed_time += SIM.ts_control
        return self.estimated_state

    def f(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        # system dynamics for propagation model: xdot = f(x, u)
        ##### TODO #####
        # pos   = x[0:3]
        vel_body = x[3:6]
        Theta = x[6:9]
        bias = x[9:12]
        # wind = np.array([[x.item(12), x.item(13), 0]]).T
        y_gyro = u[0:3]
        y_accel = u[3:6]
        g_vect = np.array([[0, 0, -MAV.gravity]]).T
        # calculate rotation matrix
        R = euler_to_rotation(Theta.item(0), Theta.item(1), Theta.item(2))
        # calculate body frame accelerations
        pos_dot = R @ vel_body
        vel_dot = cross(vel_body) @ (y_gyro - bias) + y_accel + R.T @ g_vect   
        Theta_dot = S(Theta) @ (y_gyro - bias)   
        bias_dot = np.zeros((3,1))
        wind_dot = np.zeros((2,1))  
        xdot = np.concatenate((pos_dot, vel_dot, Theta_dot, bias_dot, wind_dot), axis=0)
        return xdot

    def h_analog(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        ##### TODO #####
        # analog sensor measurements and pseudo measurements
        pos = x[0:3]
        vel_body = x[3:6]
        Theta = x[6:9]
        #bias = x[9:12]
        wind = np.array([[x.item(12), x.item(13), 0]]).T
        abs_pres = -MAV.rho * MAV.gravity * pos.item(2)
        R = euler_to_rotation(Theta.item(0), Theta.item(1), Theta.item(2))
        Va = (vel_body - R.T @ wind)
        Va2 = Va.T @ Va
        diff_pres = 0.5 * MAV.rho * Va2
        diff_pres = diff_pres.item(0)
        sideslip = np.array([[0, 1, 0]]) @ Va
        sideslip = sideslip.item(0)
    
        y = np.array([[abs_pres, diff_pres, sideslip]]).T
        return y

    def h_gps(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        ##### TODO #####        
        # measurement model for gps measurements
        pos = x[0:3]
        vel_body = x[3:6]
        Theta = x[6:9]

        R = euler_to_rotation(Theta.item(0), Theta.item(1), Theta.item(2))

        pn = pos.item(0)
        pe = pos.item(1)
        P = np.block([np.eye(2), np.zeros((2,1))])
        vg_perp = P @ R @ vel_body
        Vg = np.linalg.norm(vg_perp)
        chi = np.arctan2(vg_perp.item(1), vg_perp.item(0))

        y = np.array([[pn, pe, Vg, chi]]).T
        return y

    def h_pseudo(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        ##### TODO ##### 
        # measurement model for wind triangle pseudo measurement
        #pos = x[0:3]
        vel_body = x[3:6]
        Theta = x[6:9]
        psi = Theta.item(2)
        #bias = x[9:12]

        R = euler_to_rotation(Theta.item(0), Theta.item(1), Theta.item(2))
        
        wind = np.array([[x.item(12), x.item(13), 0]]).T
        wn = wind.item(0)
        we = wind.item(1)
        Va = (vel_body - R.T @ wind)
        Va = Va.item(0)

        P = np.block([np.eye(2), np.zeros((2,1))])
        vg_perp = P @ R @ vel_body
        Vg = np.linalg.norm(vg_perp)
        chi = np.arctan2(vg_perp.item(1), vg_perp.item(0))

        y = np.array([
            [Va * np.cos(psi) + wn - Vg * np.cos(chi)],  # wind triangle x
            [Va * np.sin(psi) + we - Vg * np.sin(chi)],  # wind triangle y
        ])
        return y


def to_MsgState(x: np.ndarray) -> MsgState:
    state = MsgState()
    state.north = x.item(0)
    state.east = x.item(1)
    state.altitude = -x.item(2)
    vel_body = x[3:6]
    state.phi = x.item(6)
    state.theta = x.item(7)
    state.psi = x.item(8)
    state.bx = x.item(9)
    state.by = x.item(10)
    state.bz = x.item(11)
    state.wn = x.item(12)
    state.we = x.item(13)
    # estimate needed quantities that are not part of state
    R = euler_to_rotation(
        state.phi,
        state.theta,
        state.psi)
    vel_world = R @ vel_body
    wind_world = np.array([[state.wn], [state.we], [0]])
    wind_body = R.T @ wind_world
    vel_rel = vel_body - wind_body
    state.Va = np.linalg.norm(vel_rel)
    state.alpha = np.arctan(vel_rel.item(2) / vel_rel.item(0))
    state.beta = np.arcsin(vel_rel.item(1) / state.Va)
    state.Vg = np.linalg.norm(vel_world)
    state.chi = np.arctan2(vel_world.item(1), vel_world.item(0))
    return state