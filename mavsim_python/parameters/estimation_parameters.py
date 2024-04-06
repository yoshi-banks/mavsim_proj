import numpy as np
import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR

# -------- EKF parameters --------
Q = np.diag([
    (10e-4)**2,  # pn
    (10e-4)**2,  # pe
    (10e-4)**2,  # pd
    (10e-4)**2,  # u
    (10e-4)**2,  # v
    (10e-4)**2,  # w
    (10e-8)**2,  # phi
    (10e-8)**2,  # theta
    (10e-8)**2,  # psi
    (10e-4)**2,  # bx
    (10e-4)**2,  # by
    (10e-4)**2,  # bz
    (10e-4)**2,  # wn
    (10e-4)**2,  # we
    ])
P0= np.diag([
    .001**2,  # pn
    .001**2,  # pe
    .001**2,  # pd
    .001**2,  # u
    .001**2,  # v
    .001**2,  # w
    np.radians(.001)**2,  # phi
    np.radians(.001)**2,  # theta
    np.radians(.001)**2,  # psi
    np.radians(.001)**2,  # bx
    np.radians(.001)**2,  # by
    np.radians(.001)**2,  # bz
    .001**2,  # wn
    .001**2,  # we
    ])

xhat0=np.array([[
    MAV.north0,  # pn
    MAV.east0,  # pe
    MAV.down0,  # pd
    MAV.u0,  # u
    MAV.v0,  # v
    MAV.w0,  # w
    MAV.phi0,  # phi
    MAV.theta0,  # theta
    MAV.psi0,  # psi
    0,  # bx
    0,  # by
    0,  # bz
    0,  # wn
    0,  # we
    ]]).T

Qu=np.diag([
    SENSOR.gyro_sigma**2, 
    SENSOR.gyro_sigma**2, 
    SENSOR.gyro_sigma**2, 
    SENSOR.accel_sigma**2,
    SENSOR.accel_sigma**2,
    SENSOR.accel_sigma**2])

R_analog = np.diag([
    SENSOR.abs_pres_sigma**2,
    SENSOR.diff_pres_sigma**2,
    (0.01)**2
])

R_gps = np.diag([
    SENSOR.gps_n_sigma**2,
    SENSOR.gps_e_sigma**2,
    SENSOR.gps_Vg_sigma**2,
    SENSOR.gps_course_sigma**2
])

R_pseudo = np.diag([
            (0.0316)**2,  # pseudo measurement #1
            (0.0316)**2,  # pseudo measurement #2
            ])
