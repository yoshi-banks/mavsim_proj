
######################################################################################
                #   sample times, etc
######################################################################################
import numpy as np
ts_simulation = 0.01  # smallest time step for simulation
start_time = 0.  # start time for simulation
end_time = 400.  # end time for simulation

ts_plot_refresh = 2  # seconds between each plot update
ts_plot_record_data = 0.1 # seconds between each time data is recorded for plots
ts_video = 0.1  # write rate for video
ts_control = ts_simulation  # sample rate for the controller


north_max = 300.0  # maximum north position
east_max = 300.0  # maximum east position
altitude_max = 100.0  # maximum altitude
phi_max = np.radians(45.0)  # maximum roll angle
theta_max = np.radians(45.0)  # maximum pitch angle
psi_max = np.radians(45.0)  # maximum yaw angle
Va_max = 10.0  # maximum velocity
alpha_max = np.radians(10.0)  # maximum angle of attack
beta_max = np.radians(10.0)  # maximum sideslip angle
p_max = np.radians(10.0)  # maximum roll rate
q_max = np.radians(10.0)  # maximum pitch rate
r_max = np.radians(10.0)  # maximum yaw rate
Vg_max = 10.0  # maximum GPS velocity
gamma_max = np.radians(10.0)  # maximum airspeed angle
chi_max = np.radians(10.0)  # maximum course angle
wn_max = 10.0  # maximum wind speed in the north direction
we_max = 10.0  # maximum wind speed in the east direction
bx_max = np.radians(10.0)  # maximum gyro bias along roll axis
by_max = np.radians(10.0)  # maximum gyro bias along pitch axis
bz_max = np.radians(10.0)  # maximum gyro bias along yaw axis
camera_az_max = np.radians(10.0)  # maximum camera azimuth angle
camera_el_max = np.radians(10.0)  # maximum camera elevation angle
