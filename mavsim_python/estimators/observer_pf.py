import numpy as np
from estimators.observer_base import ObserverBase
from estimators.filter_library.src.particle_filter import ParticleFilter
import parameters.aerosonde_parameters as MAV
import parameters.estimation_parameters_pf as EST
import parameters.sensor_parameters as SENSOR
from message_types.msg_sensors import MsgSensors
from message_types.msg_state import MsgState
from scipy.stats import multivariate_normal
from estimators.filter_library.src.utils import rk4
from tools.wrap import wrap
from scipy import stats


class ObserverPF(ObserverBase):

    def __init__(self, t0: float, initial_measurements: MsgSensors=MsgSensors()):
        super().__init__(initial_measurements)

        self.t_minus = t0

        x0_attitude = EST.xhat0_attitude
        P0_attitude = EST.P0_attitude

        x0_position = EST.xhat0_position
        P0_position = EST.P0_position

        pfParams = {'K': 0.2}
        x0_attitude_pd = multivariate_normal(x0_attitude.flatten(), P0_attitude, allow_singular=True)
        N = 10

        x0_position_pd = multivariate_normal(x0_position.flatten(), P0_position, allow_singular=True)

        # create the particle filters
        self.attitude_filter = ParticleFilter(x0_attitude_pd, N)

        self.position_filter = ParticleFilter(x0_position_pd, N)

        attitude_process_noise_pdf = multivariate_normal(np.zeros(2), EST.Q_attitude, allow_singular=True)

        position_process_noise_pdf = multivariate_normal(np.zeros(7), EST.Q_position, allow_singular=True)

        self.attitudeParams = {'dt': EST.ts_simulation}
        self.attitudeAccelStepParams = {
            'fdis': self.fdis_attitude,
            'h': self.h_accel,
            'R': EST.R_accel,
            'process_noise_pdf': attitude_process_noise_pdf,
            'K': EST.K_attitude_accel,
        }

        self.positionParams = {'dt': EST.ts_simulation}
        self.positionStepParams = {
            'fdis': self.fdis_smooth,
            'hlist': [self.h_pseudo, self.h_gps],
            'Rlist': [EST.R_pseudo, EST.R_gps],
            'process_noise_pdf': position_process_noise_pdf,
            'Klist': [EST.K_position_pseudo, EST.K_position_gps],
        }

    def fdis_attitude(self, x: np.ndarray, u: np.ndarray, omega, t, params) -> np.ndarray:
        '''
        
        '''
        dt = params['dt']
        simfunc = lambda tin, xin: self.f_attitude(xin, u) + omega
        xnext = rk4(simfunc, t, dt, x)
        return xnext
    
    def fdis_smooth(self, x: np.ndarray, u: np.ndarray, omega, t, params) -> np.ndarray:
        '''
        
        '''
        dt = params['dt']
        simfunc = lambda tin, xin: self.f_smooth(xin, u) + omega
        xnext = rk4(simfunc, t, dt, x)
        return xnext
    
    def h_accel(self, x: np.ndarray, u: np.ndarray, omega, t, params)->np.ndarray:
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
    
    def h_pseudo(self, x: np.ndarray, u: np.ndarray, omega, t, params)->np.ndarray:
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
    
    def h_gps(self, x: np.ndarray, u: np.ndarray, omega, t, params)->np.ndarray:
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

    def update(self, measurement: MsgSensors, t) -> MsgState:
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
        y_accel=np.array([
                [measurement.accel_x],
                [measurement.accel_y],
                [measurement.accel_z],
                ])
        
        xhat_attitude, P_attitude = self.attitude_step(
            y=y_accel, 
            u=u_attitude,
            t=t)
        
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
        y_pseudo = np.array([[0.], [0.]])
        
        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):
            y_gps = np.array([
                    [measurement.gps_n],
                    [measurement.gps_e],
                    [measurement.gps_Vg],
                    [wrap(measurement.gps_course, self.estimated_state.chi)],
                    ])
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

            xhat_position, P_position = self.position_step(
                y=[y_pseudo, y_gps], 
                u=u_smooth,
                t=t)
        else:
            xhat_position, P_position = self.position_step(
                y=[y_pseudo], 
                u=u_smooth,
                t=t)



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
        
    def attitude_step(self, y, u, t):
        self.attitude_filter.step(y, u, t, self.attitudeParams, self.attitudeAccelStepParams)
        return self.attitude_filter.x, self.attitude_filter.P
    
    def position_step(self, y, u, t):
        self.position_filter.multi_step(y, u, t, self.positionParams, self.positionStepParams)
        return self.position_filter.x, self.position_filter.P