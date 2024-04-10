import numpy as np
from scipy import stats
import parameters.sensor_parameters as SENSOR
import parameters.aerosonde_parameters as MAV
import parameters.simulation_parameters as SIM

# -------- EKF parameters for observer --------
Q_attitude=np.diag([
    (10e-5)**2, # phi
    (10e-5)**2, # theta
    ])
P0_attitude= np.diag([
    (.01)**2, # phi
    (.01)**2, # theta
    ])
xhat0_attitude=np.array([
    [MAV.phi0], # phi 
    [MAV.theta0], # theta
    ])
Qu_attitude=np.diag([
    SENSOR.gyro_sigma**2, 
    SENSOR.gyro_sigma**2, 
    SENSOR.gyro_sigma**2, 
    SENSOR.abs_pres_sigma])

Q_position=np.diag([
    (10e-2)**2,  # pn
    (10e-2)**2,  # pe
    (10e-4)**2,  # Vg
    (10e-4)**2, # chi
    (10e-4)**2, # wn
    (10e-4)**2, # we
    (10e-4)**2, # psi
    ])
P0_position=np.diag([
    (.01)**2, # pn
    (.01)**2, # pe
    (.01)**2, # Vg
    np.radians(.01)**2, # chi
    (.01)**2, # wn
    (.01)**2, # we
    (.01*np.pi/180.)**2, # psi
    ])
xhat0_position=np.array([
    [MAV.north0], # pn 
    [MAV.east0], # pe 
    [np.linalg.norm(np.array([[MAV.u0, MAV.v0, MAV.w0]]))], # Vg 
    [np.arctan2(MAV.v0, MAV.u0)], # chi
    [0.0], # wn 
    [0.0], # we 
    [MAV.psi0], # psi
    ])
Qu_position=0.*np.diag([
    SENSOR.gyro_sigma**2, 
    SENSOR.gyro_sigma**2, 
    SENSOR.abs_pres_sigma,
    np.radians(3), # guess for noise on roll
    np.radians(3), # guess for noise on pitch
    ])

R_accel = np.diag([
        SENSOR.accel_sigma**2, 
        SENSOR.accel_sigma**2, 
        SENSOR.accel_sigma**2
        ])
accel_gate_threshold = stats.chi2.isf(q=0.01, df=3)
R_pseudo = np.diag([
        0.01,  # pseudo measurement #1 ##### TODO #####
        0.01,  # pseudo measurement #2 ##### TODO #####
        ])
pseudo_gate_threshold = stats.chi2.isf(q=0.01, df=2)
R_gps = np.diag([
        SENSOR.gps_n_sigma**2,  # y_gps_n
        SENSOR.gps_e_sigma**2,  # y_gps_e
        SENSOR.gps_Vg_sigma**2,  # y_gps_Vg
        SENSOR.gps_course_sigma**2,  # y_gps_course
        ])

ts_simulation = SIM.ts_simulation