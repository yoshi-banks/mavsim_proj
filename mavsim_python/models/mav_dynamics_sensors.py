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
from message_types.msg_sensors import MsgSensors
import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
from models.mav_dynamics_control import MavDynamics as MavDynamicsNoSensors
from tools.rotations import quaternion_to_rotation, quaternion_to_euler, euler_to_rotation

class MavDynamics(MavDynamicsNoSensors):
    def __init__(self, Ts):
        super().__init__(Ts)
        # initialize the sensors message
        self._sensors = MsgSensors()
        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.  # large value ensures gps updates at initial time.

    def sensors(self):
        "Return value of sensors on MAV: gyros, accels, absolute_pressure, dynamic_pressure, GPS"
       
        # simulate rate gyros(units are rad / sec)
        p = self.true_state.p
        q = self.true_state.q
        r = self.true_state.r
        eta_gyro_x = np.random.randn() * SENSOR.gyro_sigma
        eta_gyro_y = np.random.randn() * SENSOR.gyro_sigma
        eta_gyro_z = np.random.randn() * SENSOR.gyro_sigma
        self._sensors.gyro_x = p + SENSOR.gyro_x_bias + eta_gyro_x
        self._sensors.gyro_y = q + SENSOR.gyro_y_bias + eta_gyro_y
        self._sensors.gyro_z = r + SENSOR.gyro_z_bias + eta_gyro_z

        # simulate accelerometers(units of g)
        fx = self._forces.item(0)
        fy = self._forces.item(1)
        fz = self._forces.item(2)
        mass = MAV.mass
        phi = self.true_state.phi
        theta = self.true_state.theta
        psi = self.true_state.psi
        eta_accel_x = np.random.randn() * SENSOR.accel_sigma
        eta_accel_y = np.random.randn() * SENSOR.accel_sigma
        eta_accel_z = np.random.randn() * SENSOR.accel_sigma
        self._sensors.accel_x = fx / mass + MAV.gravity * np.sin(theta) + eta_accel_x
        self._sensors.accel_y = fy / mass - MAV.gravity * np.cos(theta) * np.sin(phi) + eta_accel_y
        self._sensors.accel_z = fz / mass - MAV.gravity * np.cos(theta) * np.cos(phi) + eta_accel_z

        # simulate magnetometers
        # magnetic field in provo has magnetic declination of 12.5 degrees
        # and magnetic inclination of 66 degrees
        R_mag = euler_to_rotation(0, -SENSOR.mag_inclination, SENSOR.mag_declination)
        mag_intertial = R_mag @ np.array([[1,0,0]]).T
        R = euler_to_rotation(phi, theta, psi)
        mag_body = R @ mag_intertial
        self._sensors.mag_x = mag_body.item(0) + np.random.normal(SENSOR.mag_beta, SENSOR.mag_sigma)
        self._sensors.mag_y = mag_body.item(1) + np.random.normal(SENSOR.mag_beta, SENSOR.mag_sigma)
        self._sensors.mag_z = mag_body.item(2) + np.random.normal(SENSOR.mag_beta, SENSOR.mag_sigma)

        # simulate pressure sensors
        B_abs = 0
        B_diff = 0
        eta_abs = np.random.randn() * SENSOR.abs_pres_sigma
        eta_diff = np.random.randn() * SENSOR.diff_pres_sigma
        self._sensors.abs_pressure = MAV.rho * MAV.gravity * self.true_state.altitude + B_abs + eta_abs
        self._sensors.diff_pressure = MAV.rho * self.true_state.Va**2 / 2 + B_diff + eta_diff
        
        # simulate GPS sensor
        if self._t_gps >= SENSOR.ts_gps:
            self._gps_eta_n = np.exp(-SENSOR.gps_k * SENSOR.ts_gps) * self._gps_eta_n + SENSOR.ts_gps * np.random.normal(0, SENSOR.gps_n_sigma)
            self._gps_eta_e = np.exp(-SENSOR.gps_k * SENSOR.ts_gps) * self._gps_eta_e + SENSOR.ts_gps * np.random.normal(0, SENSOR.gps_e_sigma)
            self._gps_eta_h = np.exp(-SENSOR.gps_k * SENSOR.ts_gps) * self._gps_eta_h + SENSOR.ts_gps * np.random.normal(0, SENSOR.gps_h_sigma)
            self._sensors.gps_n = self.true_state.north + self._gps_eta_n
            self._sensors.gps_e = self.true_state.east + self._gps_eta_e
            self._sensors.gps_h = self.true_state.altitude + self._gps_eta_h
            Vn = self.true_state.Va * np.cos(self.true_state.psi) + self.true_state.wn
            Ve = self.true_state.Va * np.sin(self.true_state.psi) + self.true_state.we
            self._sensors.gps_Vg = np.sqrt(Vn**2 + Ve**2) + np.random.normal(0, SENSOR.gps_Vg_sigma)
            self._sensors.gps_course = np.arctan2(Ve, Vn) + np.random.normal(0, SENSOR.gps_course_sigma)
            self._t_gps = 0.
        else:
            self._t_gps += self._ts_simulation
        return self._sensors

    def external_set_state(self, new_state):
        self._state = new_state

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
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
        self.true_state.bx = SENSOR.gyro_x_bias
        self.true_state.by = SENSOR.gyro_y_bias
        self.true_state.bz = SENSOR.gyro_z_bias
        self.true_state.camera_az = self._state.item(13)
        self.true_state.camera_el = self._state.item(14)