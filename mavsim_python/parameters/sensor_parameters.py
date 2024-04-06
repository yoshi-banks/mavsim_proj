import numpy as np

noise = True

if noise:
    #-------- Accelerometer --------
    accel_sigma = 0.0025*9.81  # standard deviation of accelerometers in m/s^2

    #-------- Rate Gyro --------
    gyro_x_bias = np.radians(5*np.random.uniform(-1, 1))  # bias on x_gyro
    gyro_y_bias = np.radians(5*np.random.uniform(-1, 1))  # bias on y_gyro
    gyro_z_bias = np.radians(5*np.random.uniform(-1, 1))  # bias on z_gyro
    gyro_sigma = np.radians(0.13)  # standard deviation of gyros in rad/sec

    #-------- Pressure Sensor(Altitude) --------
    abs_pres_sigma = 0.01*1000  # standard deviation of absolute pressure sensors in Pascals
    abs_pres_bias = 0 

    #-------- Pressure Sensor (Airspeed) --------
    diff_pres_sigma = 0.002*1000  # standard deviation of diff pressure sensor in Pascals
    diff_pres_bias = 0

    #-------- Magnetometer --------
    mag_beta = np.radians(1.0)
    mag_sigma = np.radians(0.03)
    # magnetic field in provo has magnetic declination of 12.5 degrees
    # and magnetic inclination of 66 degrees
    mag_inclination = np.radians(66) 
    mag_declination = np.radians(12.5)

    # #-------- GPS --------
    # ts_gps = 1.0
    # gps_k = 1. / 1100.  # 1 / s
    # gps_n_sigma = 0.21
    # gps_e_sigma = 0.21
    # gps_h_sigma = 0.40
    # gps_Vg_sigma = 0.05
    # gps_course_sigma = gps_Vg_sigma / 10

    #-------- 2017 GPS --------
    ts_gps = 0.2
    gps_k = 1. / 1100.  # 1 / s
    gps_n_sigma = 0.01
    gps_e_sigma = 0.01
    gps_h_sigma = 0.03
    gps_Vg_sigma = 0.005
    gps_course_sigma = gps_Vg_sigma / 20
else:
    #-------- Accelerometer --------
    accel_sigma = 0.001 # standard deviation of accelerometers in m/s^2

    #-------- Rate Gyro --------
    gyro_x_bias = 0.00  # bias on x_gyro
    gyro_y_bias = 0.00  # bias on y_gyro
    gyro_z_bias = 0.00  # bias on z_gyro
    gyro_sigma = 0.001  # standard deviation of gyros in rad/sec

    #-------- Pressure Sensor(Altitude) --------
    abs_pres_sigma = 0.001  # standard deviation of absolute pressure sensors in Pascals
    abs_pres_bias = 0 

    #-------- Pressure Sensor (Airspeed) --------
    diff_pres_sigma = 0.001  # standard deviation of diff pressure sensor in Pascals
    diff_pres_bias = 0

    #-------- Magnetometer --------
    mag_beta = 0
    mag_sigma = 0.001
    # magnetic field in provo has magnetic declination of 12.5 degrees
    # and magnetic inclination of 66 degrees
    mag_inclination = np.radians(66) 
    mag_declination = np.radians(12.5)

    # #-------- GPS --------
    # ts_gps = 1.0
    # gps_k = 1. / 1100.  # 1 / s
    # gps_n_sigma = 0.21
    # gps_e_sigma = 0.21
    # gps_h_sigma = 0.40
    # gps_Vg_sigma = 0.05
    # gps_course_sigma = gps_Vg_sigma / 10

    #-------- 2017 GPS --------
    ts_gps = 0.2
    gps_k = 1. / 1100.  # 1 / s
    gps_n_sigma = 0
    gps_e_sigma = 0
    gps_h_sigma = 0
    gps_Vg_sigma = 0
    gps_course_sigma = gps_Vg_sigma / 20

